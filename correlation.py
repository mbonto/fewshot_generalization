import torch
import numpy as np
import random
import collections
from sklearn.cluster import KMeans
from sklearn import metrics
import argparse

from toolbox import load_pickle, LR_classifier, shift_operator, eigenvalues, global_ratio
from toolbox import sample_case, l2_norm, compute_confidence_interval, diffused


def get_features(model):
    assert model in ['wideresnet', 'densenet-t']
    out_dict = load_pickle(f'./features/{model}/test.pkl')
    return out_dict


def semi_supervised_dataset(out_dict, n_shot, n_way, n_query):
    # Pick a random run
    train, test, trainLabels, testLabels = sample_case(out_dict, n_shot, n_way, n_query)
    
    # Normalization
    train = l2_norm(train)      
    test = l2_norm(test)

    # Permutation
    permuted_indexes = np.random.permutation(len(train))
    train = train[permuted_indexes]
    trainLabels = trainLabels[permuted_indexes]
    
    permuted_indexes = np.random.permutation(len(test))
    test = test[permuted_indexes]
    testLabels = testLabels[permuted_indexes]
    
    # Semi-supervised setting:
    semiSupLabels = torch.cat((trainLabels.clone(), testLabels))  
    semiSup = torch.cat((train.clone(), test))
    
    return train, semiSup, trainLabels, semiSupLabels


def supervised_dataset(out_dict, n_shot, n_way, n_query):
    # Pick a random run
    train, test, trainLabels, testLabels = sample_case(out_dict, n_shot, n_way, n_query)
        
    # Normalization
    train = l2_norm(train)      
    test = l2_norm(test)
    
    # Permutation
    permuted_indexes = np.random.permutation(len(train))
    train = train[permuted_indexes]
    trainLabels = trainLabels[permuted_indexes]
    
    permuted_indexes = np.random.permutation(len(test))
    test = test[permuted_indexes]
    testLabels = testLabels[permuted_indexes]
    
    return train, test, trainLabels, testLabels


def generate_unbalanced(data, labels, n_way, n_shot, n_query, p): 
    # Retrieve supervised data
    supData = data[:n_way*n_shot]
    supLabels = labels[:n_way*n_shot]
    
    # Generate unbalance
    unbalancedData = torch.FloatTensor()
    unbalancedLabels = torch.LongTensor()
    
    # Total number of samples
    nos = n_query * n_way
    
    # Number of samples in the unbalanced class
    nosMain = int(p*nos)
    # Equal number of samples in the other classes
    nosOther = int((nos - nosMain)/(n_way - 1))
    
    nos = {}
    nos[0] = nosMain
    for w in range(1, n_way):
        nos[w] = nosOther
        
    for w in range(n_way):
        newData = data[n_way*n_shot:][labels[n_way*n_shot:]==w][:nos[w]]
        newLabels = labels[n_way*n_shot:][labels[n_way*n_shot:]==w][:nos[w]]
        
        unbalancedData = torch.cat((unbalancedData, newData), dim=0)
        unbalancedLabels = torch.cat((unbalancedLabels, newLabels), dim=0)
    
    # Permute the unsupervised samples
    permuted_indexes = np.random.permutation(len(unbalancedData))
    unbalancedData = unbalancedData[permuted_indexes]
    unbalancedLabels = unbalancedLabels[permuted_indexes]
    
    # Add the supervised samples
    unbalancedData = torch.cat((supData, unbalancedData), dim=0)
    unbalancedLabels = torch.cat((supLabels, unbalancedLabels), dim=0)

    return unbalancedData, unbalancedLabels


def get_correlation_unsup(n_way, n_shot, n_query, n_neighbor, n_run, p):
    # In these setting, we consider n_shot + n_query unlabeled samples per class. The only interest in
    # doing that, is easily comparing the correlations with the ones obtained in a semi-supervised setting.
    
    stat = collections.defaultdict(list)

    for run in range(n_run):
        print(f'Run {run}', end='\r')
        # Pick a random run, balanced or unbalanced
        if p is not None:  # unbalanced dataset
            ## We consider a dataset with n_query * n_way query samples.
            ## In order to generate an unbalanced dataset, we need to generate n_query * n_way per class.    
            _, semiSup, _, semiSupLabels = semi_supervised_dataset(out_dict, n_shot, n_way, n_query*n_way)
            semiSup, semiSupLabels = generate_unbalanced(semiSup, semiSupLabels, n_way, n_shot, n_query, p)
        else:  # balanced dataset    
            _, semiSup, _, semiSupLabels = semi_supervised_dataset(out_dict, n_shot, n_way, n_query)
            
        # Compute metrics and generalization performance
        ## Features are first diffused on a cosine similarity graph whose vertices are labeled and unlabeled samples.
        graph = shift_operator(semiSup, removeSelfConnections=True, laplacian=False,
                               nNeighbor=min(n_way*(n_shot+n_query), n_neighbor))   
        diffusedTrain = diffused(semiSup, graph, alpha=0.75, kappa=1)
        
        egv = eigenvalues(diffusedTrain, 'cosine', min(n_way*(n_shot+n_query), n_neighbor))

        km = KMeans(n_clusters=n_way)
        kmLabels = km.fit_predict(diffusedTrain.numpy())
        DBScore =  metrics.davies_bouldin_score(diffusedTrain.numpy(), kmLabels)
        ARI = metrics.adjusted_rand_score(semiSupLabels, kmLabels)
        
        # Store the results
        stat['ARI'].append(ARI)
        stat['DBScore'].append(DBScore)
        stat[f'egv'].append(egv[n_way-1].item())

    # Remind the setting
    print(f'Unsupervised setting -- {n_way}-way {n_shot}-shot {n_query}-query tasks')
    if p is not None:
        print(f'\t Unbalanced number of unlabeled samples par class: p = {p}')
    # Print the average ARI and 95% interval
    mean, std = compute_confidence_interval(stat['ARI'])
    print(f'ARI on the query samples: {np.round(mean, 2)} +- {np.round(std, 2)}')
    # Print the correlations
    print('Correlations between metrics on the unlabeled samples and ARI:')
    ## DBScore
    corr = np.abs(np.round(np.corrcoef(stat['DBScore'], stat['ARI'])[0, 1], 2))
    print(f'DBScore: {corr}')
    ## egv
    corr = np.abs(np.round(np.corrcoef(stat['egv'], stat['ARI'])[0, 1], 2))
    print(f'{n_way}-th eigenvalue: {corr}')
    

def get_correlation_semi(n_way, n_shot, n_query, n_neighbor, n_run, p):
    stat = collections.defaultdict(list)

    for run in range(n_run):
        print(f'Run {run}', end='\r')
        # Pick a random run, balanced or unbalanced
        if p is not None:  # unbalanced dataset
            ## We consider a dataset with n_query * n_way query samples.
            ## In order to generate an unbalanced dataset, we need to generate n_query * n_way per class.  
            train, semiSup, trainLabels, semiSupLabels = semi_supervised_dataset(out_dict, n_shot, n_way, n_query*n_way)
            semiSup, semiSupLabels = generate_unbalanced(semiSup, semiSupLabels, n_way, n_shot, n_query, p)
        else:  # balanced dataset    
            train, semiSup, trainLabels, semiSupLabels = semi_supervised_dataset(out_dict, n_shot, n_way, n_query)
        
        # Compute metrics and generalization performance
        ## Features are first diffused on a cosine similarity graph whose vertices are labeled and unlabeled samples
        graph = shift_operator(semiSup, removeSelfConnections=True, laplacian=False,
                               nNeighbor=min(n_way*(n_shot+n_query), n_neighbor))   
        diffusedTrain = diffused(semiSup, graph, alpha=0.75, kappa=1)
        
        _, _, LRTrainLoss, _, LRTestAcc, _, LRTestConfidence = LR_classifier(
            diffusedTrain[:n_way*n_shot], trainLabels, diffusedTrain[n_way*n_shot:],
            semiSupLabels[n_way*n_shot:], n_way)
        
        similarity = global_ratio(diffusedTrain[:n_way*n_shot], trainLabels, 'cosine')
        
        egv = eigenvalues(diffusedTrain, 'cosine', min(n_way*(n_shot+n_query), n_neighbor))

        km = KMeans(n_clusters=n_way)
        kmLabels = km.fit_predict(diffusedTrain.numpy())
        DBScore =  metrics.davies_bouldin_score(diffusedTrain.numpy(), kmLabels)       
        
        # Store the results
        stat['LRTestAcc'].append(LRTestAcc)
        stat['LRTrainLoss'].append(LRTrainLoss)
        stat['LRTestConfidence'].append(LRTestConfidence)
        stat['similarity'].append(similarity)
        stat['DBScore'].append(DBScore)
        stat[f'egv'].append(egv[n_way-1].item())

    # Remind the setting
    print(f'Semi-supervised setting -- {n_way}-way {n_shot}-shot {n_query}-query tasks')
    if p is not None:
        print(f'\t Unbalanced number of unlabeled samples par class: p = {p}')
    # Print the average accuracy and 95% interval
    mean, std = compute_confidence_interval(stat['LRTestAcc'])
    print(f'Average LR accuracy on the query samples: {np.round(mean, 2)} +- {np.round(std, 2)}')
    # Print the correlations
    print('Correlations between metrics on the training samples (labeled and unlabeled) and LR accuracy:')
    ## LRTrainLoss
    corr = np.abs(np.round(np.corrcoef(stat['LRTrainLoss'], stat['LRTestAcc'])[0, 1], 2))
    print(f'LR loss on the training samples: {corr}')
    ## similarity
    corr = np.abs(np.round(np.corrcoef(stat['similarity'], stat['LRTestAcc'])[0, 1], 2))
    print(f'Similarity: {corr}')
    ## DBScore
    corr = np.abs(np.round(np.corrcoef(stat['DBScore'], stat['LRTestAcc'])[0, 1], 2))
    print(f'DBScore: {corr}')
    ## egv
    corr = np.abs(np.round(np.corrcoef(stat['egv'], stat['LRTestAcc'])[0, 1], 2))
    print(f'{n_way}-th eigenvalue: {corr}')
    ## LRTestConfidence
    corr = np.abs(np.round(np.corrcoef(stat['LRTestConfidence'], stat['LRTestAcc'])[0, 1], 2))
    print(f'LR confidence on the query samples: {corr}')
    
    
def get_correlation_super(n_way, n_shot, n_query, n_neighbor, n_run):
    stat = collections.defaultdict(list)
    
    for run in range(n_run):
        print(f'Run {run}', end='\r')
        # Pick a random run
        train, test, trainLabels, testLabels = supervised_dataset(out_dict, n_shot, n_way, n_query)
        
        # Compute metrics and generalization performance
        _, _, LRTrainLoss, _, LRTestAcc, _, _ = LR_classifier(train, trainLabels, test, testLabels, n_way)

        similarity = global_ratio(train, trainLabels, 'cosine')

        egv = eigenvalues(train, 'cosine', min(n_way*n_shot, n_neighbor))

        km = KMeans(n_clusters=n_way)
        kmLabels = km.fit_predict(train.numpy())
        if n_shot != 1:
            DBScore =  metrics.davies_bouldin_score(train.numpy(), kmLabels)
        else:
            DBScore = None

        # Store the results
        stat['LRTestAcc'].append(LRTestAcc)
        stat['LRTrainLoss'].append(LRTrainLoss)
        stat['similarity'].append(similarity)
        stat['DBScore'].append(DBScore)
        stat[f'egv'].append(egv[n_way-1].item())
            
    # Remind the setting
    print(f'Supervised setting -- {n_way}-way {n_shot}-shot {n_query}-query tasks')
    # Print the average accuracy and 95% interval
    mean, std = compute_confidence_interval(stat['LRTestAcc'])
    print(f'Average LR accuracy on the query samples: {np.round(mean, 2)}% +- {np.round(std, 2)}%\n')
    # Print the correlations
    print('Correlations between metrics on the training samples and LR accuracy')
    ## LRTrainLoss
    corr = np.abs(np.round(np.corrcoef(stat['LRTrainLoss'], stat['LRTestAcc'])[0, 1], 2))
    print(f'LR loss on the training samples: {corr}')
    ## similarity
    corr = np.abs(np.round(np.corrcoef(stat['similarity'], stat['LRTestAcc'])[0, 1], 2))
    print(f'Similarity: {corr}')
    ## DBScore
    if n_shot != 1:
        corr = np.abs(np.round(np.corrcoef(stat['DBScore'], stat['LRTestAcc'])[0, 1], 2))
        print(f'DBScore: {corr}')
    else:
        print(f'DBScore: non applicable')
    ## egv
    corr = np.abs(np.round(np.corrcoef(stat['egv'], stat['LRTestAcc'])[0, 1], 2))
    print(f'{n_way}-th eigenvalue: {corr}')
    

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating correlations between metrics and generalization performances.')
    parser.add_argument('--model', default='wideresnet', type=str, help='backbone: wideresnet or densenet-t')
    parser.add_argument('--setting', default='supervised', type=str, help='supervised, semi-supervised or unsupervised')
    parser.add_argument('--n_way', default=5, type=int, help='number of classes')
    parser.add_argument('--n_shot', default=5, type=int, help='number of training samples')
    parser.add_argument('--n_query', default=5, type=int, help='number of test examples, which is also the number of additional unlabeled samples in some settings.')
    parser.add_argument('--n_neighbor', default=15, type=int, help='number of nearest neigbors kept in the cosine similarity graph')
    parser.add_argument('--p', type=float, help='indicate the imbalance number of unlabeled samples par class: None or float number between 0 and 1. None means no imbalance, float indicates the proportion of samples in one class with respect to the other classes.')
    parser.add_argument('--n_run', default=1000, type=int, help='number of tasks')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed = 0
    random.seed = 0
    torch.cuda.seed = 0

    # Retrieve features
    out_dict = get_features(args.model)

    # Print the correlations
    assert args.setting in ['supervised', 'semi-supervised', 'unsupervised']

    if args.setting == 'supervised':
        get_correlation_super(args.n_way, args.n_shot, args.n_query, args.n_neighbor, args.n_run)
    elif args.setting == 'semi-supervised':
        get_correlation_semi(args.n_way, args.n_shot, args.n_query, args.n_neighbor, args.n_run, args.p)
    else:
        get_correlation_unsup(args.n_way, args.n_shot, args.n_query, args.n_neighbor, args.n_run, args.p)





