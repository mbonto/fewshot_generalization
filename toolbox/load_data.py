import torch
import numpy as np
import random


def sample_case(out_dict, n_shot, n_way, n_query):
    sample_class = random.sample(list(out_dict.keys()), n_way)
  
    train_set, test_set  = [], []
    train_label, test_label = [], []
    
    for classe, each_class in enumerate(sample_class):
        samples = random.sample(out_dict[each_class], n_shot + n_query)
        train_set.append(samples[:n_shot])
        test_set.append(samples[n_shot:])
        train_label.append([classe] * n_shot)
        test_label.append([classe] * n_query)
        
    train_set = np.concatenate(train_set, axis=0)
    test_set = np.concatenate(test_set, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    
    train_set = torch.FloatTensor(train_set)
    test_set = torch.FloatTensor(test_set)
    train_label = torch.LongTensor(train_label)
    test_label = torch.LongTensor(test_label)

    return train_set, test_set, train_label, test_label


def l2_norm(data_set):
    """Divide each feature vector by its l2 norm."""
    data_set = data_set / (torch.norm(data_set, dim=1).unsqueeze(dim=1) + 0.0001)
    return data_set


