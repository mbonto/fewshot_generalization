import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_lr():
    return 0.01


def get_n_epoch():
    return 50


class logistic_regression(nn.Module):
    def __init__(self, n_feature, n_way):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_feature, n_way)
        
    def forward(self, inputs):
        outputs = self.linear(inputs)  # softmax computed via CrossEntropyLoss
        return outputs


def train_logistic_regression(data, labels):
    """ Return a logistic regression trained on data/labels.
    Parameters:
        data -- torch.tensor (n_sample, n_feature), contains training data
        labels -- torch.tensor (n_sample, 1), contains training labels
    """
    # Hyperparameters
    n_epoch = get_n_epoch()
    lr = get_lr()
    weight_decay = 5e-6
    batch_size = 5
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a classifier
    n_way = (torch.max(labels) + 1).item()  # number of classes
    classifier = logistic_regression(data.shape[1], n_way)
    classifier.to(device)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    # Stats
    loss_history = []

    for epoch in range(n_epoch):
        steps = data.shape[0] // batch_size
        sum_loss = 0
        total = 0

        permut = np.random.permutation(data.shape[0])
        data = data[permut]
        labels = labels[permut]
        
        for step in range(steps):            
            batch_data = data[batch_size*step:batch_size*(step+1)].to(device)
            batch_label = labels[batch_size*step:batch_size*(step+1)].to(device)
            # Forward
            optimizer.zero_grad()
            batch_output = classifier(batch_data)
            # Backward
            loss = criterion(batch_output, batch_label)
            loss.backward()
            optimizer.step()
            # History
            sum_loss += loss.detach().cpu().item()
            total += batch_label.shape[0]
        
        loss_history.append(sum_loss / total)

    return loss_history, classifier, criterion
 

def LR_classifier(train_set, train_labels, test_set, test_labels, n_way):
    """ Wrapper function which trains a logistic regression on train_set/train_labels and return the accuracy on test_set.
    Parameters:
        train_set -- torch.tensor (n_sample, n_feature), contains training data
        train_labels -- torch.tensor (n_sample, 1), contains training labels
        test_set  -- torch.tensor(n_test, n_feature), contains test data
        test_labels  -- torch.tensor(n_test, 1), contains test labels
        n_way -- number of classes
    """ 
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train a logistic regression
    loss_per_epoch, classifier, criterion = train_logistic_regression(train_set, train_labels)
    
    # Set model to evaluate mode
    classifier.eval()
    
    # Check the performances on the train set (accuracy, loss and confidence)
    correct = 0
    total = 0
    train_loss = 0
    train_confidence = 0
    for inputs, labels in zip(train_set, train_labels):
        inputs = inputs.unsqueeze(dim=0).to(device)
        labels = labels.unsqueeze(dim=0).to(device)
        # Model
        with torch.no_grad():        
            outputs = classifier(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)

            correct += (predicted_labels == labels).sum()
            total += labels.shape[0]
            train_loss += criterion(outputs, labels)
            conf, _ = torch.max(F.softmax(outputs.data, dim=1), 1)
            train_confidence += conf            
            
    train_accuracy = 100. * correct.item() / total  # be careful: rounding problem by doing 100. * correct / total
    train_loss = train_loss.item() / total
    train_confidence = train_confidence.item() / total
    
    # Evaluate the performances on the test set (accuracy, loss and confidence)
    correct = 0
    total = 0
    test_loss = 0
    test_confidence = 0
    for inputs, labels in zip(test_set, test_labels):
        inputs = inputs.unsqueeze(dim=0).to(device)
        labels = labels.unsqueeze(dim=0).to(device)
        # Model
        with torch.no_grad():        
            outputs = classifier(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)

            correct += (predicted_labels == labels).sum()
            total += labels.shape[0]
            test_loss += criterion(outputs, labels)
            conf, _ = torch.max(F.softmax(outputs.data, dim=1), 1)
            test_confidence += conf
            
    test_accuracy = 100. * correct.item() / total
    test_loss = test_loss.item() / total
    test_confidence = test_confidence.item() / total
    
    return loss_per_epoch, train_accuracy, train_loss, train_confidence, test_accuracy, test_loss, test_confidence

