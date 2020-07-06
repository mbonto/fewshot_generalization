import numpy as np


def compute_confidence_interval(data):
    """
    Return mean and 95% confidence interval of values in a list
    
    Parameter:
        data -- A list of values.
    """
    a = np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(a.shape[0]))
    return m, pm
