import time
import numpy as np
import random
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pylab as plt
from tqdm import tqdm
from tqdm import trange
import sys

class VariationDescent:
    '''
    Класс для решения вариационного неравенства
    '''

    def __init__(self, func, nabla_f, x_0, y_0, gamma_k, args, max_steps=400,
                 z_sol=None, eps=1e-6, criterium='x_k - x^*', use_proj=False, proj_func=None,
                 use_extragrad=False, use_smp=False, use_mbfw=False):
        self.func = func
        self.gradient = nabla_f
        self.x_0 = x_0
        self.y_0 = y_0
        self.gamma_k = gamma_k
        self.args = args
        self.max_steps = max_steps
        self.z_sol = z_sol
        self.eps = eps
        self.criterium = criterium
        self.use_proj = use_proj
        self.proj_func = proj_func
        self.use_extragrad = use_extragrad
        self.use_smp = use_smp
        self.use_mbfw = use_mbfw

    def get_next_default(self, x_curr, y_curr, k):
        '''
        Градиентный спуск(подъем)
        '''

        gamma = self.gamma_k(k, self.args)

        if self.args['sega'] is True:
            self.args['batch_size'] = 1
            e_i = np.zeros(self.args['d'])
            grad_x, grad_y = self.gradient(x_curr, y_curr, self.args)
            i = self.args['i']
            e_i[i] = 1.
            h_x_next = self.args['h_x'] + (grad_x[i] - self.args['h_x'][i]) * e_i
            grad_x_next = self.args['d'] * (grad_x[i] - self.args['h_x'][i]) * e_i + self.args['h_x']
            self.args['grad_x_curr'] = np.copy(grad_x_next)
            self.args['h_x'] = np.copy(h_x_next)
            h_y_next = self.args['h_y'] + (grad_y[i] - self.args['h_y'][i]) * e_i
            grad_y_next = self.args['d'] * (grad_y[i] - self.args['h_y'][i]) * e_i + self.args['h_y']
            self.args['grad_y_curr'] = np.copy(grad_y_next)
            self.args['h_y'] = np.copy(h_y_next)
            grad_x = grad_x_next
            grad_y = grad_y_next
        else:
            grad_x, grad_y = self.gradient(x_curr, y_curr, self.args)

        x_next = x_curr - gamma * grad_x
        y_next = y_curr + gamma * grad_y

        return x_next, y_next

    def get_next_extragrad(self, x_curr, y_curr, x_tmp, y_tmp, k):
        '''
        Метод Экстраградиента
        '''
        gamma = self.gamma_k(k, self.args)


        x_tmp_next = x_curr - gamma * self.nabla_f_x(x_tmp, y_tmp, self.args)
        y_tmp_next = y_curr + gamma * self.nabla_f_y(x_tmp, y_tmp, self.args)
        x_next = x_curr - gamma * self.nabla_f_x(x_tmp_next, y_tmp_next, self.args)
        y_next = y_curr + gamma * self.nabla_f_y(x_tmp_next, y_tmp_next, self.args)

        return x_next, y_next, x_tmp_next, y_tmp_next

    def get_next_smp(self, r_x_curr, r_y_curr, k):
        '''
        Метод SMP
        '''
        def prox(z, xi):
            ret = np.zeros(len(z))
            tmp = 0
            for z_i, xi_i in zip(z, xi):
                tmp += z_i * np.exp(-xi_i)

            for j, (z_j, xi_j) in enumerate(zip(z, xi)):
                ret[j] = (1./tmp) * z_j * np.exp(-xi_j)

            return ret

        gamma = self.gamma_k(k, self.args)

        w_x_curr = prox(r_x_curr, gamma * self.nabla_f_x(r_x_curr, r_y_curr, self.args))
        w_y_curr = prox(r_y_curr, -gamma * self.nabla_f_y(r_x_curr, r_y_curr, self.args))
        r_x_next = prox(r_x_curr, gamma * self.nabla_f_x(w_x_curr, w_y_curr, self.args))
        r_y_next = prox(r_y_curr, -gamma * self.nabla_f_y(w_x_curr, w_y_curr, self.args))

        return r_x_next, r_y_next, w_x_curr, w_y_curr, gamma
    
    def get_next_mbfw(self, x, x_previous, y, y_previous, k):
        """
        Momentum-Based Frank-Wolfe
        """
        learning_rate = self.gamma_k(k, self.args)

        momentum = self.args['momentum_k'](k, self.args)
        if self.args['sega'] is True:
            self.args['batch_size'] = 1
            grad_x_prev = np.copy(self.args['grad_x_curr'])
            grad_y_prev = np.copy(self.args['grad_y_curr'])
            self.args['batch_size'] = 1
            e_i = np.zeros(self.args['d'])
            grad_x, grad_y = self.gradient(x, y, self.args)
            i = self.args['i']
            e_i[i] = 1.
            h_x_next = self.args['h_x'] + (grad_x[i] - self.args['h_x'][i]) * e_i
            grad_x_next = self.args['d'] * (grad_x[i] - self.args['h_x'][i]) * e_i + self.args['h_x']
            self.args['grad_x_curr'] = np.copy(grad_x_next)
            self.args['h_x'] = np.copy(h_x_next)
            h_y_next = self.args['h_y'] + (grad_y[i] - self.args['h_y'][i]) * e_i
            grad_y_next = self.args['d'] * (grad_y[i] - self.args['h_y'][i]) * e_i + self.args['h_y']
            self.args['grad_y_curr'] = np.copy(grad_y_next)
            self.args['h_y'] = np.copy(h_y_next)
        else:
            grad_x_next, grad_y_next = self.gradient(x, y, self.args)
            grad_x_prev, grad_y_prev = self.gradient(x_previous, y_previous, self.args)

        z_x = self.args['z_x_k']
        z_y = self.args['z_y_k']
        z_x_k = (1 - momentum) * z_x + momentum * grad_x_next + (1 - momentum) * \
                (grad_x_next - grad_x_prev)
        z_y_k = (1 - momentum) * z_y + momentum * grad_y_next + (1 - momentum) * \
                (grad_y_next - grad_y_prev)
        self.args['z_x_k'] = z_x_k
        self.args['z_y_k'] = z_y_k

        i_min = np.argmin(z_x_k)
        s_x_k = np.zeros(len(x), dtype=float)
        s_x_k[i_min] = 1.
        j_max = np.argmax(z_y_k)
        s_y_k = np.zeros(len(x), dtype=float)
        s_y_k[j_max] = 1.

        x_next = x + learning_rate * (s_x_k - x)
        y_next = y + learning_rate * (s_y_k - y)

        return x_next, y_next
    
    def search(self):
        """
        :return: x_curr - значение аргумента, на котором достигается минимум функции
        :return: errors - вектор значений критерия на каждой итерации
        :return: times - вектор времен работы
        """
        z_0 = np.hstack([self.x_0, self.y_0])
        if self.criterium == 'z_k - z^*':
            errors = [norm(z_0 - self.z_sol, ord=2)]
        elif self.criterium == 'Err_vi':
            errors = [np.max(self.args['A'] @ self.x_0) - np.min(self.args['A'].T @ self.y_0)]
        else:
            raise ValueError('Wrong criterium! Only \n z_k - z^* \n Err_vi \n are available!')

        x_curr = np.copy(self.x_0)
        y_curr = np.copy(self.y_0)
        x_tmp = np.copy(self.x_0)
        y_tmp = np.copy(self.y_0)
        r_x_curr = np.copy(self.x_0)
        r_y_curr = np.copy(self.y_0)
        gamma_list = []
        w_x_list = []
        w_y_list = []
        x_prev = np.copy(self.x_0)
        y_prev = np.copy(self.y_0)
        self.args['h_x'], self.args['h_y'] = self.gradient(self.x_0, self.y_0, self.args)
        self.args['grad_x_curr'] = np.copy(self.args['h_x'])
        self.args['grad_y_curr'] = np.copy(self.args['h_y'])
        start_time = time.time()
        times = [0]
        for k in trange(self.max_steps, file=sys.stdout, colour="green"):
            self.args['k'] = k + 1
            # шаг градиентного спуска
            if self.use_extragrad is True:
                x_next, y_next, x_tmp, y_tmp = VariationDescent.get_next_extragrad(self, x_curr,
                                                                                   y_curr,
                                                                                   x_tmp, y_tmp, k)
            elif self.use_smp is True:
                r_x_curr, r_y_curr, w_x_curr, w_y_curr, gamma = VariationDescent.get_next_smp(self,
                                                                                              r_x_curr,
                                                                                              r_y_curr, k)
                gamma_list.append(gamma)
                w_x_list.append(w_x_curr)
                w_y_list.append(w_y_curr)
                x_next = 1./sum(gamma_list) * sum([a * b for a, b in zip(w_x_list, gamma_list)])
                y_next = 1. / sum(gamma_list) * sum([a * b for a, b in zip(w_y_list, gamma_list)])
            elif self.use_mbfw is True:
                x_next, y_next = VariationDescent.get_next_mbfw(self, x_curr, x_prev, y_curr, y_prev, k)
            else:
                x_next, y_next = VariationDescent.get_next_default(self, x_curr, y_curr, k)

            if self.use_proj is True:
                x_next = self.proj_func(x_next, self.args)
                x_tmp = self.proj_func(x_tmp, self.args)
                y_next = self.proj_func(y_next, self.args)
                y_tmp = self.proj_func(y_tmp, self.args)

            # добавление новой ошибки в вектор errors
            error = None
            z_next = np.hstack([x_next, y_next])
            if self.criterium == 'z_k - z^*':
                error = norm(z_next - self.z_sol, ord=2)
            elif self.criterium == 'Err_vi':
                error = np.max(self.args['A'] @ x_next) - np.min(self.args['A'].T @ y_next)

            time_now = time.time()
            errors.append(error)
            times.append(time_now - start_time)

            x_prev = x_curr
            x_curr = x_next
            y_prev = y_curr
            y_curr = y_next

            # критерий остановки
            if error <= self.eps:
                break

        iterations = np.array(range(1, len(errors) + 1))
        errors = np.array(errors)
        times = np.array(times)
        return x_curr, y_curr, iterations, errors, times
    
def get_grad_tpf_jaguar(x, y, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    batch_size = args['batch_size']
    d = args['d']
    nabla_f_x = np.zeros(d, dtype=float)
    nabla_f_y = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = [-1, 1][random.randrange(2)]
        ########
        args['i'] = i
        ########
        nabla_f_x += (func(x + gamma * e, y, args) - \
                      func(x - gamma * e, y, args)) / (2. * gamma) * e
        
        nabla_f_y += (func(x, y + gamma * e, args) - \
                      func(x, y - gamma * e, args)) / (2. * gamma) * e

        args['oracle_counter'] += 4

    return (float(d) / float(batch_size)) * nabla_f_x, (float(d) / float(batch_size)) * nabla_f_y


def get_grad_tpf_lame_v2(x, y, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    for i in range(d):
        e = np.zeros(d, dtype=float)
        e[i] = 1
        
        nabla_f_x += (func(x + gamma * e, y, args) - \
                      func(x - gamma * e, y, args)) / (2. * gamma) * e
        
        nabla_f_y += (func(x, y + gamma * e, args) - \
                      func(x, y - gamma * e, args)) / (2. * gamma) * e

        args['oracle_counter'] += 4

    return nabla_f_x, nabla_f_y


def get_grad_tpf_lame_v1(x, y, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f_x = np.zeros(d, dtype=float)
    nabla_f_y = np.zeros(d, dtype=float)

    for _ in range(batch_size):
        np.random.seed(random.randint(1, 10000))
        e = 2 * np.random.rand(d) - 1
        e = e / np.linalg.norm(e, ord=norm)
        
        nabla_f_x += (func(x + gamma * e, y, args) - func(x - gamma * e, y, args)) / (2. * gamma) * e
        nabla_f_y += (func(x, y + gamma * e, args) - func(x, y - gamma * e, args)) / (2. * gamma) * e

        args['oracle_counter'] += 4

    return 1. /batch_size * nabla_f_x, 1. /batch_size * nabla_f_y


def get_grad_opf_lame(x, y, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f_x = np.zeros(d, dtype=float)
    nabla_f_y = np.zeros(d, dtype=float)
    
    for _ in range(batch_size):
        np.random.seed(random.randint(1, 10000))
        e = 2 * np.random.random_sample(d) - 1
        e = e / np.linalg.norm(e, ord=norm)
        
        nabla_f_x += d * func(x + gamma * e, y, args) * e / gamma
        nabla_f_y += d * func(x, y + gamma * e, args) * e / gamma

        args['oracle_counter'] += 2

    return 1./batch_size * nabla_f_x, 1./batch_size * nabla_f_y


def get_grad_opf_jaguar(x, y, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    batch_size = args['batch_size']
    d = args['d']
    nabla_f_x = np.zeros(d, dtype=float)
    nabla_f_y = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = [-1, 1][random.randrange(2)]
        ########
        args['i'] = i
        ########
        nabla_f_x += func(x + gamma * e, y, args) * e / gamma
        nabla_f_y += func(x, y + gamma * e, args) * e / gamma

        args['oracle_counter'] += 2

    return nabla_f_x, nabla_f_y