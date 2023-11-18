import numpy as np
from importlib import reload

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

    return args