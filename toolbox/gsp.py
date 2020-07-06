import torch


def gft(x):
    # Check whether laplacian is symmetric
    assert (x == x.T).all()
    # Compute eigenvalues, eigenvectors
    # v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]
    w, v = torch.symeig(x, eigenvectors=True)
    return w, v


def similarity(data1, data2, metric, alpha=1): 
    eps = 0.0001
    if metric == 'cosine':
        norm1 = torch.norm(data1, dim=-1).unsqueeze(dim=-1)
        norm2 = torch.norm(data2, dim=-1).unsqueeze(dim=-1)
        data1 = data1 / (norm1 + eps)
        data2 = data2 / (norm2 + eps)
        S = torch.matmul(data1, torch.transpose(data2, dim0=-2, dim1=-1))
    else:
        raise NotImplementedError(f'Unknown metric {metric}')
    return S


def degree(graph):
    D = torch.sum(graph, dim=1)
    return D


def laplacian(graph, lplcn):
    """
    Parameters:
        lplcn -- 'combinatorial'or 'normalized'
    """
    D = degree(graph)  # in-degree matrix
    if lplcn == 'combinatorial':
        L = torch.diag(D) - graph
    elif lplcn == 'normalized':
        D = D ** (-1/2)
        D = torch.diag(D)
        L = torch.mm(torch.mm(D, graph), D)
        L = torch.eye(graph.shape[0], device=L.device) - L
    else:
        raise NotImplementedError("Wrong laplacian name.")
    return L


