import collections
from math import ceil
import torch
import torch.nn as nn
import numpy as np
import pygsp
from tqdm import tqdm
from utils import get_device
import scipy


class logistic_regression(nn.Module):

    def __init__(self, n_feature, n_way):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_feature, n_way)

    def forward(self, inputs):
        outputs = self.linear(inputs)  # softmax computed via CrossEntropyLoss
        return outputs


def get_optimizer_myriam(classifier, epoch, n_epoch):
    lr = 10
    if epoch >= (1/3)*n_epoch:
        lr *= 0.1
    if epoch >= (2/3)*n_epoch:
        lr *= 0.1
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    return optimizer

def get_optimizer_xuqing(classifier):
    lr = 0.01
    return torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=5e-6)

def train_logistic_regression(data, labels, n_way, device):
    """ Return a trained logistic regression"""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = logistic_regression(data.shape[1], n_way)
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    n_steps = 100
    batch_size = 5
    loss_history = []
    steps_per_epoch = int(ceil(data.shape[0] / batch_size))
    n_epoch = n_steps // steps_per_epoch
    for epoch in tqdm(range(n_epoch), leave=False):
        # optimizer = get_optimizer_myriam(classifier, epoch, n_epoch)
        optimizer = get_optimizer_xuqing(classifier)
        permut = np.random.permutation(data.shape[0])
        data = data[permut]
        labels = labels[permut]
        sum_loss = 0
        for step in range(steps_per_epoch):
            start_batch, end_batch = batch_size*step, batch_size*(step+1)
            inputs = data[start_batch:end_batch].to(device)
            label = labels[start_batch:end_batch].to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            sum_loss += loss
        loss_history.append(sum_loss.detach().cpu().item())
    return classifier


def evaluate_logistic_regression(classifier, input_set, labels_set, device):
    correct, total = 0, 0
    for inputs, labels in zip(input_set, labels_set):
        inputs = inputs.unsqueeze(dim=0).to(device)
        labels = labels.unsqueeze(dim=0).to(device)
        with torch.no_grad():
            results = classifier(inputs)
            _, predicted = torch.max(results.data, 1)
            total += int(labels.shape[0])
            correct += int((predicted == labels).sum().item())
    accuracy = 100. * correct / total
    return accuracy


def normalize_train_test(train_set, test_set, mode):
    if 'mean' in mode:
        train_mean = torch.mean(train_set, dim=0, keepdim=True)
        train_set = train_set - train_mean
        test_set = test_set - train_mean
    if 'std' in mode:
        epsilon = 0.  #1e-4
        train_std = epsilon + torch.std(train_set, dim=0, keepdim=True)
        train_set = train_set / train_std
        test_set = test_set / train_std
    if 'l2' in mode:
        train_set = train_set / torch.norm(train_set, dim=1, keepdim=True)
        test_set = test_set / torch.norm(test_set, dim=1, keepdim=True)
    if 'l1' in mode:
        train_set = train_set / torch.sum(train_set, dim=1, keepdim=True)
        test_set = test_set / torch.sum(test_set, dim=1, keepdim=True)
    return train_set, test_set


def nearest_centroid_classifier(train_set, train_labels, test_set, test_labels, metric):
    # Compute the means of the feature vectors of the same classes
    n_way = torch.max(train_labels) + 1
    means = torch.zeros(n_way, train_set.shape[1])
    for label in range(n_way):
        means[label] = torch.mean(train_set[train_labels==label], dim=0)
    if metric == 'cosine':
        means = means / torch.norm(means, dim=1, keepdim=True)
    accs = []
    for data_set, labels in zip([train_set, test_set], [train_labels, test_labels]):
        # Compute the similarity of the query feature vectors with respect to the means  test_set.shape[0], n_way
        if metric == 'cosine':
            data = data_set / torch.norm(data_set, dim=1, keepdim=True)
            similarities = torch.mm(data, torch.transpose(means, dim0=0, dim1=1))
        elif metric == 'euclidean':
            similarities = torch.cdist(data_set, means)
            similarities = torch.exp(-1 * similarities)
        # Choose the labels according to the closest mean
        predicted = torch.argmax(similarities, dim=1)
        # Compute accuracy
        total = labels.shape[0]
        correct = (predicted == labels).sum()
        acc = 100. * correct / total
        accs.append(acc.item())
    return accs

def tikhonov_label_propagation(train_set, train_labels, test_set, test_labels, params, loss):
    import diffusion_graph
    import baseline_graph
    import monitoring
    loss_fn = baseline_graph.cosine_loss
    x_latent = torch.cat([train_set, test_set])
    weights = diffusion_graph.weights_from_loss_fn(x_latent, loss_fn,
                                                   params.num_neighbors, regular=True,
                                                   undirected=True, normalize_weights=True)
    graph = pygsp.graphs.Graph(adjacency=weights.numpy())
    labels = torch.cat([train_labels, test_labels]).numpy()
    n_shot, n_val = int(train_labels.shape[0]), int(test_labels.shape[0])
    mask = np.array([True]*n_shot + [False]*n_val)
    logits = pygsp.learning.classification_tikhonov(graph, np.copy(labels), mask, tau=0)
    prediction = np.argmax(logits, axis=1)
    return monitoring.get_acc(prediction, logits, labels, n_shot, n_val, loss)

def get_default_cherry_params():
    args = parse_args(from_command_line=False)
    dict_grid = dict(vars(args))
    Parameter = collections.namedtuple('Parameter', ' '.join(sorted(dict_grid.keys())))
    return Parameter(**dict_grid)

def features_classification(train_set, train_labels, test_set, test_labels, n_way, classifier, normalize, params, loss=False):
    if classifier == 'logistic_regression':
        train_set, test_set = normalize_train_test(train_set, test_set, normalize)
        device = get_device()
        with torch.enable_grad():
            classifier = train_logistic_regression(train_set, train_labels, n_way, device)
        classifier.eval()
        train_accuracy = evaluate_logistic_regression(classifier, train_set, train_labels, device)
        test_accuracy = evaluate_logistic_regression(classifier, test_set, test_labels, device)
        if loss:
            train_accuracy, test_accuracy, loss
        return train_accuracy, test_accuracy
    elif classifier == 'ncm':
        accs = nearest_centroid_classifier(train_set, train_labels, test_set, test_labels, metric='cosine')
        return accs
    elif classifier == 'tikhonov':
        accs = tikhonov_label_propagation(train_set, train_labels, test_set, test_labels, params, loss)
        return accs
    elif classifier == 'mean_shift':
        import monitoring
        accs = monitoring.raw_examples_mean_shift(params, train_set, train_labels, test_set, test_labels, loss)
        return accs
    elif classifier == 'thikonov_communities':
        import monitoring
        accs = monitoring.thikonov_communities(params, train_set, train_labels, test_set, test_labels, loss)
        return accs
    else:
        assert False
