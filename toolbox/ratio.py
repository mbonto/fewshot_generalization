import numpy as np
import collections
import torch
from .gsp import similarity

def inter_similarity(train, trainLabels, metric):
    labels = np.unique(trainLabels)
    interSimilarity = collections.defaultdict(dict)
    
    for l in labels:
        for m in labels:
            if l == m:
                interSimilarity[l][m] = 0
            else:
                s = similarity(train[trainLabels==l, :], train[trainLabels==m, :], metric='cosine')
                interSimilarity[l][m] = torch.sum(s).item() / s.shape[0] / s.shape[0]
    return interSimilarity


def intra_similarity(train, trainLabels, metric):
    labels = np.unique(trainLabels)
    intraSimilarity = collections.defaultdict(list)
    
    for l in labels:
        s = similarity(train[trainLabels==l, :], train[trainLabels==l, :], metric='cosine')
        if s.shape[0] != 1:
            intraSimilarity[l] = (torch.sum(s).item() - s.shape[0]) / s.shape[0] / (s.shape[0] - 1)
        # If there is only one sample in a class, the intra_similarity of the class is set to 1.
        else:
            intraSimilarity[l] = 1.0
    return intraSimilarity


# We assume we have only one cluster per class in the feature space.
def global_ratio(train, trainLabels, metric):
    ratio = 0
    interSimilarity = inter_similarity(train, trainLabels, metric)  # n_class x n_class with 0 in the diagonal
    intraSimilarity = intra_similarity(train, trainLabels, metric)  # n_class
    for label in interSimilarity.keys():
        ratio += intraSimilarity[label] - max(interSimilarity[label].values())
    ratio /= len(interSimilarity.keys())
    return ratio