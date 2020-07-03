import torch
import torch.nn as nn
import numpy as np


def cosine_loss(z_latent, bad_edges, reduce=True, memory_light=False):
    epsilon = 1e-2
    scalars = torch.einsum('ik,jk->ij', z_latent, z_latent)
    norms = epsilon + torch.norm(z_latent, dim=-1, keepdim=False)
    norms = torch.einsum('i,j->ij', norms, norms)
    cosines = scalars / norms
    losses = []
    for (a, b) in bad_edges:
        delta = cosines[a,b]
        if memory_light:
            losses.append(float(delta.item()))
        else:
            losses.append(delta)
    if reduce:
        return torch.mean(torch.stack(losses))
    return losses
