import argparse
from .utils import GridSearch

def get_grid_search_params(mode, classifier, n_way, n_val, n_shot, dataset, dot_name, arena):
    grid = GridSearch()
    grid.add_range('dataset', [dataset])
    grid.add_range('mode', [mode])
    grid.add_range('n_way', [n_way])
    grid.add_range('n_val', [n_val])
    grid.add_range('n_shot', [n_shot])
    if mode == 'monitoring_volume':
        grid.add_range('num_neighbors', [20])
        grid.add_range('regular', [False])
        grid.add_range('higher_order', [False])
        grid.add_range('kappa', [2]) # [1, 2, 3]
        grid.add_range('alpha', [0.75])  # [1.25]
        if arena:
            grid.add_range('intersection_measure', ['arena'])
        else:
            grid.add_range('intersection_measure', ['louvain_dendrogram'])
        # grid.add_range('intersection_measure', ['minimum_cut'])
        # grid.add_range('intersection_measure', ['stoer_wagner'])
        # grid.add_range('intersection_measure', ['kernighan'])
        # grid.add_range('intersection_measure', ['baseline'])
        grid.add_range('communities', ['entropy'])  # 'pure'
        grid.add_range('worse_only', [False])
        grid.add_range('parts', [64])
        grid.add_range('crop', [False])  # 'pure'
        grid.add_range('ladder', [-1])  # 'pure'
        grid.add_range('dot_name', [dot_name])
    grid.add_range('origin_normalization', ['mean-l2'])
    grid.add_range('latent_normalization', ['l2'])
    grid.add_range('classifier', [classifier])  # ['logistic_regression', 'ncm']
    grid.add_range('compute_corr', [True])
    grid.add_range('plot', [True])
    grid.add_range('progressive_plot', [False])
    return grid

def parse_args(modes):
    parser = argparse.ArgumentParser(description='Graph Smoothness.')
    parser.add_argument('--dataset', default='wideresnet', help='Dataset key.')
    parser.add_argument('--mode', default='monitoring_volume', help='Mode of regularization. Can be %s'%modes)
    parser.add_argument('--classifier', default='logistic_regression', help='How to classify examples.')
    parser.add_argument('--n_way', default=5, type=int, help='number of classes.')
    parser.add_argument('--n_val', default=595, type=int, help='number of validation examples.')
    parser.add_argument('--n_shot', default=5, type=int, help='number of training examples.')
    parser.add_argument('--dot_name', default='densenet-m/novel/louvain_dendrogram_communities_1_20.dot')
    parser.add_argument('--arena', action='store_true')
    return parser.parse_args()
