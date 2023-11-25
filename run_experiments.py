import numpy as np
from importlib import reload
from sklearn.datasets import load_svmlight_file

import utils
reload(utils)

def init_experiment(func_name, d, seed=18, **init_args):
    np.random.seed(seed)
    args = {}
    if func_name == "quadratic":
        L = init_args['L']
        mu = init_args['mu']
        args['A'] = utils.generate_matrix(d, mu, L)
        args['b'] = np.random.random(size=d)
    elif func_name == "mushrooms":
        dataset = "mushrooms.txt" 
        data = load_svmlight_file(dataset)
        X, y = data[0].toarray(), data[1]
        y = y * 2 - 3
        matrix = X * np.expand_dims(y, axis=1)
        args['matrix'] = matrix

    return args