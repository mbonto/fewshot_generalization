import torch
from .gsp import gft, similarity, degree
from .gsp import laplacian as laplacian_matrix


def eigenvalues(train, metric, nNeighbor):
    shiftOperator = shift_operator(train, removeSelfConnections=True, laplacian=True, nNeighbor=nNeighbor)
    shiftOperator = (shiftOperator + shiftOperator.T)/2
    w, v = gft(shiftOperator)
    return w


def knn_without_sym(graph, nNeighbor, setting):
    if setting == 'column':
        graph = graph.T
    nNeighbor = min(graph.shape[1], nNeighbor)
    kBiggest= torch.argsort(graph, 1)[:,int(-nNeighbor)]  # sort the colums, rows by rows
    # Store the weigths of the kth closest neighbour of each row of graph
    thresholds = graph[torch.arange(graph.shape[0]), kBiggest].reshape(-1,1)
    # Create adjacency_matrix
    adj = (graph >= thresholds) * 1.0 
    # Weighted adjacency matrix
    adj = adj.type(torch.float32)
    adj = adj * graph  
    if setting == 'column':
        adj = adj.T
    return adj


# Shift operator (nearest neighbors kept in row or column)
def shift_operator(datapoints, removeSelfConnections=False, laplacian=False, nNeighbor=None, setting='row'):
    nPoint = datapoints.shape[0]
    shiftOperator = similarity(datapoints, datapoints, "cosine")
    if removeSelfConnections:
        for i in range(nPoint):
            shiftOperator[i, i] = 0
    if nNeighbor:
        shiftOperator = knn_without_sym(shiftOperator, nNeighbor, setting)
    if laplacian:
        shiftOperator = (shiftOperator + shiftOperator.T)/2
        shiftOperator = laplacian_matrix(shiftOperator, "combinatorial")
    return shiftOperator


def diffused(labels, graph, alpha, kappa):
    D = degree(graph)
    D = D ** (-1/2)
    D = torch.diag(D)
    graph = torch.mm(torch.mm(D, graph), D)
    
    graph = alpha * torch.eye(graph.shape[0]) + graph
    filters = graph.clone()
    for loop in range(kappa):
        filters = torch.matmul(filters, graph)
    
    propagatedSignal = torch.matmul(filters, labels)
    return propagatedSignal

