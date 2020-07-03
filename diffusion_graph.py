import itertools
import sklearn.decomposition
import torch
import numpy as np
from latent_graph import embed_into_graph
from classifiers import train_logistic_regression
from utils import get_device
from pygsp import graphs, filters, learning


def get_item(loss):
    return float(loss.item())

def symmetrize_edges(edge_loss_pairs):
    def reverse_edge(pair):
        loss, (a, b) = pair
        return loss, (b, a)
    symmetrics = map(reverse_edge, edge_loss_pairs)
    return edge_loss_pairs + list(symmetrics)

def filter_by_neighbours(num_nodes, all_edges, loss_per_edge, num_neighbors, regular):
    edge_loss_pairs = list(zip(loss_per_edge, all_edges))
    edge_loss_pairs = symmetrize_edges(edge_loss_pairs)
    edge_loss_pairs.sort(reverse=True)  # biggest similarity first
    max_edges = len(edge_loss_pairs) if regular else (num_neighbors * num_nodes)
    neighbours = [[] for _ in range(num_nodes)]
    threshold = 2 # non singular but not interesting
    for edge_idx, pair in enumerate(edge_loss_pairs):
        _, (a, _) = pair
        if regular and len(neighbours[a]) >= num_neighbors:
            continue
        if edge_idx < max_edges or len(neighbours[a]) < threshold:
            neighbours[a].append(pair)
    return itertools.chain.from_iterable(neighbours)

def get_degree(weights):
    return torch.sum(torch.abs(weights), dim=0)  # not suitable for undirected graphs

def laplacian_from_weights(weights):
    D = get_degree(weights)
    return D - weights

def edges_from_loss_fn(z_latent, loss_fn, num_neighbors, regular, normalize_weights=False, substract_mean=False, exponent=False):
    if exponent:
        z_latent = torch.sqrt(z_latent)
    if substract_mean:
        z_latent = z_latent - torch.mean(z_latent, dim=0, keepdim=True)
    if normalize_weights:
        z_latent = z_latent / torch.norm(z_latent, dim=1, keepdim=True)
    num_nodes = int(z_latent.shape[0])
    all_edges = [(i, j) for i in range(num_nodes) for j in range(i)]
    loss_per_edge = loss_fn(z_latent, all_edges, reduce=False, memory_light=True)
    edge_loss_pairs = filter_by_neighbours(num_nodes, all_edges, loss_per_edge, num_neighbors, regular)
    return edge_loss_pairs

def weights_from_loss_fn(z_latent, loss_fn, num_neighbors, regular, undirected, normalize_weights=False):
    num_nodes = int(z_latent.shape[0])
    edge_loss_pairs = edges_from_loss_fn(z_latent, loss_fn, num_neighbors, regular, normalize_weights)
    weights = np.zeros(shape=(num_nodes, num_nodes), dtype=np.float32)
    for loss, (a, b) in edge_loss_pairs:
        weights[a, b] = loss
    weights = torch.FloatTensor(weights)
    if undirected:
        weights = 0.5*weights + 0.5*torch.t(weights)  # symmetrize
    return weights
