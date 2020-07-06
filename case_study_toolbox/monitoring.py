import collections
import itertools
from itertools import tee
from math import sqrt
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import scipy
from scipy.stats import spearmanr
import torch
import pygsp
import networkx as nx
import community as louvain
import networkx.algorithms.community as commu
from sklearn import manifold
from tqdm import tqdm
from .diffusion_graph import edges_from_loss_fn, cosine_loss
from .loaders import get_all_pairs_datasets
from .loaders import split_train_test
from .loaders import get_train_test_datasets_labels
from .loaders import get_dataset_from_datapath
from .loaders import get_labels_stats
from .classifiers import features_classification


def connect_parts_labels(recorder, graph, part_a, part_b, labels, label_0, label_1):
    recorder.record_balance(min(len(part_a), len(part_b)) / len(graph))
    a_to_0 = np.array([label_0]*len(graph))
    for node_b in part_b:
        a_to_0[node_b] = label_1
    a_to_0 = torch.LongTensor(a_to_0)
    ratio_errors = (a_to_0 == labels).sum().item()
    ratio_errors = ratio_errors / len(graph)
    ratio_errors = max(ratio_errors, 1. - ratio_errors)
    #print('\n', part_a, ' ============ ', part_b, ' ####### ', ratio_errors*100, '\n')
    return ratio_errors * 100.

def stoer_wagner_volume(recorder, graph, labels, params):
    assert params.n_way == 2
    try:
        _, (part_a, part_b) = nx.stoer_wagner(graph)
    except nx.exception.NetworkXError:
        assert params.n_shot == 1
        part_a = nx.node_connected_component(graph, 0)
        part_b = nx.node_connected_component(graph, 1)
    label_0, label_1 = torch.min(labels).item(), torch.max(labels).item()
    assert (labels == label_0).sum().item() * 2 == len(graph)
    return connect_parts_labels(recorder, graph, part_a, part_b, labels, label_0, label_1)

def min_cut_volume(recorder, graph, labels, params):
    assert params.n_shot == 1 and params.n_way == 2
    cut = nx.minimum_edge_cut(graph, 0, 1)
    graph.remove_edges_from(cut)
    part_a = nx.node_connected_component(graph, 0)
    part_b = nx.node_connected_component(graph, 1)
    label_0, label_1 = labels[0].item(), labels[1].item()
    return connect_parts_labels(recorder, graph, part_a, part_b, labels, label_0, label_1)

def kernighan(recorder, graph, labels, params):
    assert params.n_shot == 1 and params.n_way == 2
    part_a, part_b = commu.kernighan_lin.kernighan_lin_bisection(graph, max_iter=20)
    label_0, label_1 = labels[0].item(), labels[1].item()
    return connect_parts_labels(recorder, graph, part_a, part_b, labels, label_0, label_1)

def compute_pure(nodes, colors, labels):
    labels = labels.numpy()
    colors_to_labels = collections.defaultdict(set)
    labels_to_colors = collections.defaultdict(set)
    for node in nodes:
        colors_to_labels[colors[node]].add(int(labels[node]))
        labels_to_colors[int(labels[node])].add(colors[node])
    num_confusion = sum([1 for labels in colors_to_labels.values() if len(labels) > 1])
    num_splitted = sum([1 for colors in labels_to_colors.values() if len(colors) > 1])
    num_good = len(colors_to_labels) - num_confusion
    ratio = num_good / (num_good + num_confusion) * 100.
    return ratio / 100.

def get_total(bag):
    return sum([len(subbag) for subbag in bag.values()])

def compute_entropy(bag):
    total = get_total(bag)
    freq = [len(subbag)/total for subbag in bag.values()]
    freq = np.array(freq)
    h = -(freq * np.log2(freq)).sum()
    return h

def print_histo(histo, edges):
    for histo_bin in zip(edges, edges[1:], histo.tolist()):
        print('[%.3f,%.3f]=%.3f'%histo_bin, end=' ')
    print('\n')

def compute_communities_entropy(nodes, colors, labels, verbose):
    communities = collections.defaultdict(lambda: collections.defaultdict(list))
    classes = collections.defaultdict(lambda: collections.defaultdict(list))
    for node in nodes:
        com, label = colors[node], int(labels[node])
        communities[com][label].append(node)
        classes[label][com].append(node)
    com_entropy = np.array([compute_entropy(com) for com in communities.values()])
    cla_entropy = np.array([compute_entropy(cla) for cla in classes.values()])
    totals = [get_total(com) for com in communities.values()]
    avg_cla_entropy = np.mean(cla_entropy)
    avg_com_entropy = np.mean(com_entropy)
    w_com_entropy = np.average(com_entropy, weights=totals)
    hist_cla_entropy = np.histogram(cla_entropy, bins=5, density=True)
    hist_com_entropy = np.histogram(com_entropy, bins=5, density=True)
    weights = [total / sum(totals)  for total in totals]
    histo_w_com_entropy = np.histogram(com_entropy, bins=5, weights=weights, density=True)
    if verbose:
        print('')
        print('Communities %d\n'%len(communities))
        print('Entropy per class: ', avg_cla_entropy)
        print_histo(*hist_cla_entropy)
        print('Entropy per community: ', avg_com_entropy)
        print_histo(*hist_com_entropy)
        print('Entropy per community, weighted: ', w_com_entropy)
        print_histo(*histo_w_com_entropy)
    avgs = avg_cla_entropy, avg_com_entropy, w_com_entropy
    histos = hist_cla_entropy, hist_com_entropy, histo_w_com_entropy
    return [len(communities)] + list(zip(avgs, histos))

def color(dendrogram, node, level):
    key = node
    for i in range(level+1):
        key = dendrogram[level][key]
    return key

def louvain_dendrogram(recorder, graph, labels, params, print_details=False):
    dendrogram = louvain.generate_dendrogram(graph)
    colors = {node: node for node in graph}
    infos = []
    if print_details:
        print('\n', 'Level', ' ', 'Mixed Communities', ' ', 'Splitted Classes', ' ', 'Good Communities', ' ', 'Ratio')
    for level in range(len(dendrogram)):
        colors = {node: dendrogram[level][color] for node, color in colors.items()}
        if params.communities == 'pure':
            info = compute_pure(graph.nodes, colors, labels)
        elif params.communities == 'entropy':
            info = compute_communities_entropy(graph.nodes, colors, labels, verbose=False)
        infos.append(info)
    return infos

def build_community_graph(params, examples, labels, print_graph=False):
    num_neighbors, regular = params.num_neighbors, params.regular
    edges = edges_from_loss_fn(examples, cosine_loss, num_neighbors=num_neighbors,
                               regular=regular,
                               normalize_weights=True,
                               substract_mean=True,
                               exponent=True)
    if params.higher_order:
        num_nodes = len(set([int(edge[1][0]) for edge in edges]))
        adj = np.zeros(shape=(num_nodes, num_nodes))
        for w, (a, b) in edges:
            adj[a,b] = adj
        adj = np.linalg.power(params.alpha * np.eyes(N=num_nodes) + adj, params.kappa)
    edges = [(a, b, w) for w, (a, b) in edges]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    if print_graph:
        nx.draw_spring(graph)
        plt.show()
    return graph

def monitore_volume(recorder, train_set, test_set, train_labels, test_labels, params):
    examples = torch.cat([train_set, test_set], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)
    graph = build_community_graph(params, examples, labels)
    if params.intersection_measure == 'stoer_wagner':
        volume = stoer_wagner_volume(recorder, graph, labels, params)
    elif params.intersection_measure == 'minimum_cut':
        volume = min_cut_volume(recorder, graph, labels, params)
    elif params.intersection_measure == 'kernighan':
        volume = kernighan(recorder, graph, labels, params)
    elif params.intersection_measure == 'louvain_dendrogram':
        infos = louvain_dendrogram(recorder, graph, labels, params)
        return infos
    else:
        raise ValueError
    recorder.record_volume_error(volume)

def infos_per_level(graph, labels, params, verbose=True):
    dendrogram = louvain.generate_dendrogram(graph)
    colors = {node: node for node in graph}
    infos = []
    for level in range(len(dendrogram)):
        colors = {node: dendrogram[level][color] for node, color in colors.items()}
        info = compute_communities_entropy(graph.nodes, colors, labels, verbose=verbose)
        infos.append(info)
    return infos

def get_draw_options(big_graph, bipartite):
    w_max = float(np.max([w[2] for w in big_graph.edges.data('weight')]))
    colors = [w[2]/w_max for w in big_graph.edges.data('weight')]
    edge_labels = {w[:2]:('%.3f'%w[2]) for w in  big_graph.edges.data('weight')}
    if bipartite is None:
        node_color = '#A0CBE2'
    else:
        node_color = ['#1A4081' if node < bipartite else '#E37373' for node in big_graph]
    return {
        "node_color": node_color,
        "edge_color": colors,
        "width": 4,
        "edge_cmap": plt.cm.Reds,
        "with_labels": True,
    }, edge_labels

def print_big_graph(params, mean_pos, edges, avg_degree=20, tsne_pos=False):
    bipartite = 64 if '&' in params.dataset else None
    edges.sort(key=lambda t: t[2], reverse=True)
    max_edges = int(len(mean_pos) * avg_degree)
    edges = edges[:max_edges]
    big_graph = nx.Graph()
    big_graph.add_weighted_edges_from(edges)
    options, edge_labels = get_draw_options(big_graph, bipartite)
    if tsne_pos:
        mean_pos_array = np.concatenate(list(mean_pos.values()), axis=0)
        tsne = manifold.TSNE()
        print(mean_pos_array)
        mean_pos_array = tsne.fit_transform(mean_pos_array)
        pos = dict()
        for i, key in enumerate(mean_pos):
            pos[key] = mean_pos_array[i]
    elif bipartite is not None:
        part_1 = [node for node in big_graph if node < bipartite]
        pos = nx.drawing.layout.bipartite_layout(big_graph, part_1, align='horizontal')
    else:
        # pos = nx.spring_layout(big_graph)
        pos = nx.spectral_layout(big_graph)
    nx.draw_networkx_nodes(big_graph, pos=pos, **options)
    nx.draw_networkx_labels(big_graph, pos, labels={node:str(node) for node in big_graph})
    nx.draw_networkx_edge_labels(big_graph, edge_labels=edge_labels, pos=pos, font_size=8)
    name = params.intersection_measure + '_communities_' + str(-params.ladder) + '_' + str(params.num_neighbors)
    nx.drawing.nx_agraph.write_dot(big_graph, os.path.join('graphs', name+'.dot'))
    plt.savefig(os.path.join('graphs', name+'.pdf'))
    plt.clf()
    # plt.show()
    print('')

def add_class(mean_pos, label, false_label, examples, labels):
    if label in mean_pos:
        return
    class_examples = examples[labels == false_label]
    mean_pos[label] = torch.mean(class_examples, dim=0, keepdim=True).numpy()

def monitore_communities(data_path, params, num_repetitions=5):
    parts = params.parts if '&' in params.dataset else None
    all_pairs = get_all_pairs_datasets(data_path, params.n_way, params.crop, parts)
    edges = []
    mean_pos = dict()
    total_pairs = next(all_pairs)
    progress = tqdm(total=total_pairs, leave=True)
    for (examples, labels, label_a, label_b, false_a, false_b) in all_pairs:
        add_class(mean_pos, label_a, false_a, examples, labels)
        add_class(mean_pos, label_b, false_b, examples, labels)
        graph = build_community_graph(params, examples, labels)
        h_seq = []
        for _ in range(num_repetitions):
            infos = infos_per_level(graph, labels, params, verbose=False)
            ladder = max(len(infos) + params.ladder, 0)
            h_seq.append(infos[ladder][-1][0]) # last level
        h_avg = float(np.mean(h_seq))
        h_max = float(np.log2(params.n_way))
        r = h_avg / h_max
        r = min(1-1e-3, r) # crop
        weight = r # / (1 - r) # similarity score
        edges.append((label_a, label_b, weight))
        desc = ' '.join([str(label_a), str(label_b), str(weight)])
        progress.set_description(desc=desc)
        progress.update()
    progress.close()
    print_big_graph(params, mean_pos, edges)

def monitore_regression(data_path, params, num_repetitions=20):
    all_pairs = get_all_pairs_datasets(data_path, params.n_way, params.crop, params.parts)
    edges = []
    mean_pos = dict()
    for (examples, labels, label_a, label_b, false_a, false_b) in all_pairs:
        add_class(mean_pos, label_a, false_a, examples, labels)
        add_class(mean_pos, label_b, false_b, examples, labels)
        acc_seq = []
        for _ in range(num_repetitions):
            train_test = split_train_test(examples, labels, params.n_shot, params.n_val)
            train_set, train_labels, test_set, test_labels = train_test
            train_acc, test_acc = features_classification(train_set, train_labels,
                                                          test_set, test_labels,
                                                          params.n_way, params.classifier,
                                                          params.origin_normalization, params)
            acc = 100. - test_acc # error monitoring instead of accuracy
            acc_seq.append(acc) # last level
        acc_avg = float(np.mean(acc_seq))
        acc_max = float(100.)
        r = acc_avg / acc_max
        r = min(1-1e-5, r) # crop
        weight = r # / (1 - r) # similarity score
        edges.append((label_a, label_b, weight))
        print(label_a, label_b, weight, acc_avg)
    print_big_graph(params, mean_pos, edges)

def gather_edges(graph, ways):
    edges = []
    for str_edge in graph.edges.data('weight'):
        edge = int(str_edge[0]), int(str_edge[1]), float(str_edge[2])
        if edge[0] in ways and edge[1] in ways:
            edges.append(edge)
    return edges

def get_subgraph_weight(edges):
    return sum([edge[2] for edge in edges])

def get_worse_clusters(params, big_graph, data_path):
    _, original_labels = get_dataset_from_datapath(data_path)
    num_labels, n_sample_per_label = get_labels_stats(original_labels)
    combinations_iter = itertools.combinations(list(range(big_graph.number_of_nodes())), params.n_way)
    combinations = []
    for i, ways in enumerate(combinations_iter):
        real_ways = [int(original_labels[i*n_sample_per_label]) for i in ways]
        combinations.append((get_subgraph_weight(gather_edges(big_graph, real_ways)), ways))
    combinations.sort(key=(lambda t: t[0]), reverse=True)
    return combinations

def write_labels(signal, nodes):
    for node in nodes:
        if nodes[node]['label'] is None:
            continue
        signal[node] = nodes[node]['label']

def create_signal(nodes, n_way):
    num_nodes = len(nodes)
    signal = np.full(shape=(num_nodes, n_way), fill_value=(1./n_way))
    write_labels(signal, nodes)
    return signal

def bhattacharyya_dist(signal, temperature):
    sqrt_signal = np.sqrt(signal)
    bhattacharyya = np.einsum('ip,jp->ij', sqrt_signal, sqrt_signal)
    bhattacharyya = -np.log(np.maximum(bhattacharyya, epsilon_a))
    bhattacharyya = scipy.special.softmax(adj * bhattacharyya, axis=1)
    return bhattacharyya

def kl_metric(signal, temperature):
    eps = 0.001
    num_labels = int(signal.shape[1])
    signal = np.log(np.maximum(signal, eps))
    signal = scipy.special.softmax(signal * temperature, axis=1)
    h = np.sum(signal * np.log(num_labels * np.maximum(signal, eps)), axis=1)
    num_nodes = int(signal.shape[0])
    h_cross = np.zeros(shape=(num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            h_cross[i,j] = float(h[j])
    return h_cross

def graph_mean_shift(graph, n_way, num_iterations_max=1):
    signal = create_signal(graph.nodes, n_way)
    adj = nx.adjacency_matrix(graph).toarray()
    adj = adj + np.eye(N=int(adj.shape[0]))
    adj = adj + 1.*(adj != 0.)  # from [-1, 1] to [0, 2]
    adj_temperature = 1.
    adj = scipy.special.softmax(adj * adj_temperature, axis=1)
    epsilon = 0.001
    metric_temperature = 1.
    for _ in range(num_iterations_max):
        metric = kl_metric(signal, metric_temperature)
        wcoeff = np.maximum(metric, epsilon) * adj
        wcoeff = wcoeff / np.sum(wcoeff, axis=1)
        signal = wcoeff @ signal
        write_labels(signal, graph.nodes)
        signal = signal / np.sum(signal, axis=1, keepdims=True)
    return signal

def create_graph_mean_shift(params, train_set, train_labels, test_set, test_labels):
    examples = torch.cat([train_set, test_set])
    labels = torch.cat([train_labels, test_labels])
    graph = build_community_graph(params, examples, labels)
    for node in graph.nodes:
        graph.nodes[node]['label'] = None
    num_labels = len(set(train_labels.tolist()))
    for node in range(train_labels.shape[0]):
        dirac = np.zeros(shape=(num_labels))
        label = int(train_labels[node])
        dirac[label] = 1.
        graph.nodes[node]['label'] = dirac
    return graph

def get_acc(predicted, logits, labels, train_set_size, test_set_size, loss):
    train_acc = (labels[:train_set_size] == predicted[:train_set_size]).sum()
    test_acc = (labels[train_set_size:] == predicted[train_set_size:]).sum()
    train_acc *= (100. / train_set_size)
    test_acc *= (100. / test_set_size)
    if loss:
        prob = scipy.special.softmax(logits, axis=1)
        epsilon = 0.001
        h = - prob * np.log(np.maximum(prob, epsilon))
        h = np.sum(h, axis=1)
        h = np.mean(h)
        return train_acc, test_acc, float(h)
    return train_acc, test_acc

def raw_examples_mean_shift(params, train_set, train_labels, test_set, test_labels, loss):
    graph = create_graph_mean_shift(params, train_set, train_labels, test_set, test_labels)
    with np.printoptions(threshold=np.inf):
        signal = graph_mean_shift(graph, params.n_way, num_iterations_max=100)
    labels = np.concatenate([train_labels, test_labels])
    predicted = np.argmax(signal, axis=1)
    train_set_size = int(train_set.shape[0])
    test_set_size = int(test_set.shape[0])
    return get_acc(predicted, signal, labels, train_set_size, test_set_size, loss)

def get_edges_between(graph, part_a, part_b):
    edges = []
    for node in part_a:
        for other in graph.neighbors(node):
            if other in part_b:
                edges.append(graph[node][other]['weight'])
    return max(sum(edges), 0.)

def meta_graph(graph, colors):
    communities = collections.defaultdict(list)
    for node, com in colors.items():
        communities[com].append(node)
    meta = nx.Graph()
    for i, com_a in enumerate(communities):
        meta.add_node(com_a)
        for com_b in list(communities.keys())[:i]:
            part_a, part_b = communities[com_a], communities[com_b]
            edge = get_edges_between(graph, part_a, part_b)
            if edge != 0:
                meta.add_edge(com_a, com_b, weight=edge)
    return meta

def create_labels_mask_metagraph(graph, colors, labels, n_shot, n_val):
    num_communities = len(set(colors.values()))
    num_labels = len(set(labels.numpy().tolist()))
    histo = np.zeros(shape=(num_communities, num_labels), dtype=np.float32)
    mask = [False] * num_communities
    for node in range(int(labels.shape[0])):
        color = colors[node]
        label = labels[node]
        histo[color, label] += 1.
        if node < n_shot:  # training set: we are certain about this measurement
            mask[color] = True
    histo = histo / np.sum(histo, axis=1, keepdims=True)  # no null element guaranteed
    return histo, np.array(mask)

def get_examples_predicted(colors, predicted_communities):
    predicted = [None] * len(colors)
    for node in colors:
        color = colors[node]
        predicted[node] = predicted_communities[color]
    return np.array(predicted)

def thikonov_communities(params, train_set, train_labels, test_set, test_labels, loss):
    examples = torch.cat([train_set, test_set])
    labels = torch.cat([train_labels, test_labels])
    graph = build_community_graph(params, examples, labels)
    dendrogram = louvain.generate_dendrogram(graph)
    colors = {node: node for node in graph}
    max_level = 1 # len(dendrogram)
    for level in range(max_level):
        colors = {node: dendrogram[level][color] for node, color in colors.items()}
    meta = meta_graph(graph, colors)
    meta = pygsp.graphs.Graph.from_networkx(meta)
    n_shot, n_val = int(train_labels.shape[0]), int(test_labels.shape[0])
    com_labels, mask = create_labels_mask_metagraph(meta, colors, labels, n_shot, n_val)
    logits_communities = pygsp.learning.regression_tikhonov(meta, np.copy(com_labels), mask, tau=0.)
    predicted_communities = np.argmax(logits_communities, axis=1)
    predicted = get_examples_predicted(colors, predicted_communities)
    return get_acc(predicted, logits_communities, labels.numpy(), n_shot, n_val, loss)

def node_label(node):
    if node < 64:
        return node
    if node < 80:
        return node - 64
    return node-64-16

def init_plot():
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlim(0, 5)
    plt.ylim(0, 50)
    sc = ax.scatter([], [], marker='x')
    plt.xlabel('Sum of edge weights', fontsize='x-large')
    plt.ylabel('Error (%)', fontsize='x-large')
    # plt.title('Error as function of edge weights')
    fig.canvas.draw_idle()
    plt.pause(0.1)
    return fig, sc

def add_point(fig, sc, metric, accs):
    sc.set_offsets(np.c_[np.array(metric), np.array(accs)])
    fig.canvas.draw_idle()
    plt.pause(0.05)

def monitore_arena(data_path, params, num_repetitions=10000):
    path = os.path.join('graphs', params.dot_name)
    dot_graph = nx.drawing.nx_agraph.read_dot(path)
    edges = [(int(str_edge[0]), int(str_edge[1]), float(str_edge[2])) for str_edge in dot_graph.edges.data('weight')]
    edges.sort(key=lambda t: t[2], reverse=True)
    max_edges = len(edges)#//60
    edges = edges[:max_edges]
    print(edges)
    big_graph = nx.Graph()
    big_graph.add_weighted_edges_from(edges)
    bipartite = 64 if '&' in params.dataset else None
    if params.plot:
        if bipartite:
            part_1 = [node for node in big_graph if node < bipartite]
            pos = nx.drawing.layout.bipartite_layout(big_graph, part_1, align='horizontal')
        else:
            pos = nx.spring_layout(big_graph)
        options, edge_labels = get_draw_options(big_graph, bipartite)
        nx.draw_networkx_nodes(big_graph, pos=pos, **options)
        nx.draw_networkx_edges(big_graph, pos=pos, **options)
        nx.draw_networkx_labels(big_graph, pos, labels={node:str(node_label(node)) for node in big_graph})
        # nx.draw_networkx_edge_labels(big_graph, edge_labels=edge_labels, pos=pos, font_size=8)
        plt.show()
    weights = []
    accs = []
    if params.worse_only:
        clusters = get_worse_clusters(params, big_graph, data_path)
    progress = tqdm(total=num_repetitions, leave=True)
    fig, scs = init_plot()
    for repet in range(num_repetitions):
        ways = clusters[repet][1] if params.worse_only else None
        train_test, ways = get_train_test_datasets_labels(data_path, params.n_way, params.n_shot, params.n_val, ways=ways)
        train_set, train_labels, test_set, test_labels = train_test
        edges = gather_edges(big_graph, ways)
        edges_weights = get_subgraph_weight(edges)
        weights.append(edges_weights)
        train_acc, test_acc = features_classification(train_set, train_labels,
                                                      test_set, test_labels,
                                                      params.n_way, 'logistic_regression',
                                                      params.origin_normalization, params)
        error_rate = 100. - test_acc
        accs.append(error_rate)  # error rate
        if (repet+1) % 100 == 0:
            add_point(fig, scs, weights, accs)
        desc = ' '.join([str(train_acc), str(ways), str(edges_weights), str(test_acc)])
        progress.set_description(desc=desc)
        progress.update()
    all_results = np.array(list(zip(weights, accs)))
    np.savetxt(os.path.join('graphs', 'correlation.txt'), all_results, delimiter=',')
    progress.close()
    fig.canvas.draw_idle()
    fig.savefig(os.path.join('graphs', 'correlation.eps'), format='eps')
    plt.pause(10)
    # plt.close(fig)
    print('mean weight=', np.mean(weights))
    print('mean_acc=', 100-np.mean(accs))
    for func, func_name in zip([spearmanr, np.corrcoef], ['spearmanr', 'corrcoef']):
        corrcoefmatrix = func(weights, accs)
        print('weights', ' ', func_name, ' => ', corrcoefmatrix)
