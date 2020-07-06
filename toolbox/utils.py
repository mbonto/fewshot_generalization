import pickle


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)




