import numpy as np
from tqdm import tqdm
import argparse
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from monitoring import monitore_volume, monitore_communities, monitore_regression, monitore_arena
from parse_grid import get_grid_search_params, parse_args


def correlation_grid_search(n_dataset, data_path, grid_search_params):
    vars_corr = [('test_acc_orig', 'volume_error')]
    print(grid_search_params.get_constant_keys())
    print('')
    for param_num, params in enumerate(grid_search_params.get_params()):
        if params.training_type != 'test_baseline':
            print(grid_search_params.get_variable_keys(params))
        measures = ['louvain_dendrogram', 'baseline', 'arena']
        if params.mode == 'monitoring_volume' and params.intersection_measure in measures:
            if params.intersection_measure == 'louvain_dendrogram':
                monitore_communities(data_path, params)
            elif params.intersection_measure == 'baseline':
                monitore_regression(data_path, params)
            elif params.intersection_measure == 'arena':
                with torch.no_grad():
                    monitore_arena(data_path, params)

def get_num_tests(training_type):
    num_tests = dict()
    num_tests['single'] = 1
    return num_tests[training_type]

if __name__ == '__main__':
    modes = ['monitoring_volume']
    args = parse_args(modes)
    n_dataset = get_num_tests(args.training_type)
    data_paths = {'densenet-m-base':'features/densenet-m/train.pkl',
                  'densenet-m-val':'features/densenet-m/val.pkl',
                  'densenet-m-novel':'features/densenet-m/test.pkl',
                  'densenet-t-novel':'features/densenet-t/test.pkl',
                  'wideresnet-base':'features/wideresnet/train.pkl',
                  'wideresnet-val':'features/wideresnet/val.pkl',
                  'wideresnet-novel':'features/wideresnet/test.pkl'}
    data_path = '&'.join([data_paths[path] for path in args.dataset.split('&')])
    n_way = args.n_way
    n_val = args.n_val
    n_shot = args.n_shot
    to_print = args.mode, args.classifier, n_way, n_val, n_shot, data_path
    print('mode=%s classifier=%s n_way=%d n_val=%d n_shot=%d datapath=%s'%to_print)
    if args.mode in modes:
        grid_search_params = get_grid_search_params(args.training_type, args.mode, args.classifier,
                                                    n_way, n_val, n_shot, args.dataset, args.dot_name, args.arena)
        correlation_grid_search(n_dataset, data_path, grid_search_params)
    else:
        print('Bad experience name')
