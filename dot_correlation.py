import argparse
from scipy.stats import spearmanr
import networkx as nx

def get_edges(dot_graph):
    return {(min(int(str_edge[0]),int(str_edge[1])),max(int(str_edge[0]),int(str_edge[1]))):float(str_edge[2])
            for str_edge in dot_graph.edges.data('weight')}

def check_edges(dict_1, dict_2):
    for a, b in dict_1:
        assert (a, b) in dict_2

def compute_dot(graph_name_1, graph_name_2):
    graph_1 = nx.drawing.nx_agraph.read_dot(graph_name_1)
    graph_2 = nx.drawing.nx_agraph.read_dot(graph_name_2)
    edges_1 = get_edges(graph_1)
    edges_2 = get_edges(graph_2)
    check_edges(edges_1, edges_2)
    check_edges(edges_2, edges_1)
    w_1, w_2 = [], []
    for edge in edges_1:
        w_1.append(edges_1[edge])
        w_2.append(edges_2[edge])
    coeff = spearmanr(w_1, w_2)
    return coeff

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Mean Shift.')
    parser.add_argument('--graph_name_1', type=str)
    parser.add_argument('--graph_name_2', type=str)
    args = parser.parse_args()
    coeff = compute_dot(args.graph_name_1, args.graph_name_2)
    print(coeff)