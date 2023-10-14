import numpy as np
import random
from numpy.linalg import norm
from matplotlib import pylab as plt
from tqdm import trange
import sys


class GDOptimizer:
    def __init__(self, x_0, get_grad, step, function, gamma=lambda k: 1, max_oracle_calls=10000,
                 set_name=None, norma=2, criterion='f(x_k) - f(x^*)', use_jaguar=False, eps=1e-5, x_sol=None,
                 gradient_true=None, use_proj=False, function_without_noise=None):
        # outer props
        self.eps = eps
        self.max_oracle_calls = max_oracle_calls
        self.criterion = criterion
        self.get_grad = get_grad
        self.step = step
        self.d = x_0.size
        self.set = set_name
        self.use_jaguar = use_jaguar
        self.use_proj = use_proj
        self.function = function
        self.function_without_noise = function_without_noise if function_without_noise else function
        # inner props
        self.k = 0
        self.x = x_0
        self.x_previous = x_0
        self.x_sol = x_sol
        self.grad = None
        self.e_i = None
        self.grad_curr = None
        self.error = None
        self.errors = []
        self.x_next = None
        self.oracle_calls = []
        self.oracle_counter = 0
        self.gradient_true = gradient_true
        self.i = None
        self.norma = norma
        self.gamma = gamma
        self.h_next = None
        self.h, _, _, _ = self.get_grad(self.x, self.function, self.gamma, self.d, self.k, self.norma)

    def proj(self):
        if self.set is None:
            pass
        elif self.set == 'l1_ball':
            self.x_next = projection_l1(self.x_next)
        elif self.set == 'l2_ball':
            self.x_next = projection_l2(self.x_next)
        elif self.set == 'simplex':
            self.x_next = projection_simplex(self.x_next)
        else:
            raise ValueError('Wrong projection set!')

    def get_next(self):
        learning_rate = self.step(self.k)
        self.grad, iters_per_step, self.i, self.e_i = self.get_grad(self.x, self.function, self.gamma,
                                                                    self.d, self.k, self.norma)
        self.oracle_counter += iters_per_step
        if self.use_jaguar is True:
            #######################
            # self.h_next = self.h + (self.grad[self.i] - self.h[self.i]) * self.e_i
            # self.h = np.copy(self.h_next)
            # self.grad = np.copy(self.h_next)
            #######################
            self.h[self.i] = self.grad[self.i]
            self.grad = np.copy(self.h)

        return self.x - learning_rate * self.grad

    def get_error(self):
        if self.criterion == 'x_k - x^*':
            return norm(self.x - self.x_sol, ord=2)
        elif self.criterion == 'f(x_k) - f(x^*)':
            return self.function_without_noise(self.x) - self.function_without_noise(self.x_sol)
        elif self.criterion == 'x_k+1 - x_k':
            return norm(self.x - self.x_previous, ord=2)
        elif self.criterion == 'f(x_k+1) - f(x_k)':
            return np.abs(self.function_without_noise(self.x) - self.function_without_noise(self.x_previous))
        elif self.criterion == 'nabla(f)(x_k)':
            temp, _, _, _ = self.get_grad(self.x, self.function_without_noise, self.gamma, self.d, self.k, self.norma)
            return norm(temp, ord=2)
        elif self.criterion == 'gap':
            return self.gradient_true(self.x) @ self.x - np.min(self.gradient_true(self.x))
        else:
            raise ValueError('Wrong criterion!')

    def search(self):
        for self.k in trange(self.max_oracle_calls, file=sys.stdout, colour="green"):
            self.x_next = self.get_next()
            if self.use_proj:
                self.proj()
            self.x_previous = self.x
            self.x = self.x_next
            self.error = self.get_error()
            self.errors.append(self.error)
            self.oracle_calls.append(self.oracle_counter)
            if self.error <= self.eps or self.oracle_counter > self.max_oracle_calls:
                break
        return self.errors, self.oracle_calls, self.x


class CDOptimizer(GDOptimizer):
    def __init__(self, x_0, get_grad, step, function, gamma=lambda k: 1, max_oracle_calls=10000,
                 set_name=None, norma=2, criterion='f(x_k) - f(x^*)', use_jaguar=False, eps=1e-5, x_sol=None,
                 gradient_true=None, use_proj=False, function_without_noise = None):
        GDOptimizer.__init__(self, x_0, get_grad, step, function, gamma, max_oracle_calls, set_name, norma, criterion,
                             use_jaguar, eps, x_sol, gradient_true, use_proj, function_without_noise)

    def get_next(self):
        learning_rate = self.step(self.k)
        self.grad, iters_per_step, self.i, self.e_i = self.get_grad(self.x, self.function, self.gamma,
                                                                    self.d, self.k, self.norma)
        self.oracle_counter += iters_per_step

        return self.x - learning_rate * self.d * self.grad


class SEGAOptimizer(GDOptimizer):
    def __init__(self, x_0, get_grad, step, function, gamma=lambda k: 1, max_oracle_calls=10000,
                 set_name=None, norma=2, criterion='f(x_k) - f(x^*)', use_jaguar=False, eps=1e-5, x_sol=None,
                 gradient_true=None, use_proj=False, function_without_noise = None):
        GDOptimizer.__init__(self, x_0, get_grad, step, function, gamma, max_oracle_calls, set_name, norma, criterion,
                             use_jaguar, eps, x_sol, gradient_true, use_proj, function_without_noise)

    def get_next(self):
        learning_rate = self.step(self.k)
        self.grad, iters_per_step, self.i, self.e_i = self.get_grad(self.x, self.function, self.gamma,
                                                                    self.d, self.k, self.norma)
        self.oracle_counter += iters_per_step

        self.h_next = self.h + (self.grad[self.i] - self.h[self.i]) * self.e_i
        self.grad = self.h + self.d * (self.grad[self.i] - self.h[self.i]) * self.e_i
        self.h = np.copy(self.h_next)

        return self.x - learning_rate * self.grad


class DGDOptimizer(GDOptimizer):
    def __init__(self, percent):
        GDOptimizer.__init__(self)
        self.percent = percent

    def get_next(self):
        learning_rate = self.step(self.k)
        idxs = random.sample(range(self.d), self.d * self.percent // 100)
        grad_compressed = np.zeros(self.d)

        for j in idxs:
            self.grad, _, _, _ = self.get_grad(self.x, self.function, self.gamma, self.d, self.k, self.norma)
            grad_compressed[j] = self.grad[j]

        return self.x - learning_rate * grad_compressed


class MDOptimizer(GDOptimizer):
    def __init__(self):
        GDOptimizer.__init__(self)

    def get_next(self):
        learning_rate = self.step(self.k)
        sigma = 0
        self.grad, _, _, _ = self.get_grad(self.x, self.function, self.gamma, self.d, self.k, self.norma)

        for i, x_i in enumerate(self.x):
            sigma += np.exp(-learning_rate * self.grad[i]) * x_i

        x_next = np.copy(self.x)
        for i, x_i in enumerate(self.x):
            x_next[i] = (x_i / sigma) * np.exp(-learning_rate * self.grad[i])

        return x_next / np.linalg.norm(x_next, ord=1), 1


class FWOptimizer(GDOptimizer):
    def __init__(self, x_0, get_grad, step, function, gamma=lambda k: 1, max_oracle_calls=10000,
                 set_name=None, norma=2, criterion='f(x_k) - f(x^*)', use_jaguar=False, eps=1e-5, x_sol=None,
                 gradient_true=None, use_proj=False, use_momentum=False, momentum_func=lambda k: 1 / (k + 1), 
                 function_without_noise = None):
        GDOptimizer.__init__(self, x_0, get_grad, step, function, gamma, max_oracle_calls, set_name, norma, 
                             criterion, use_jaguar, eps, x_sol, gradient_true, use_proj, function_without_noise)
        self.h_next = None
        self.h, _, _, _ = self.get_grad(self.x, self.function, self.gamma, self.d, self.k, self.norma)
        self.y_next = None
        self.y = np.zeros(self.d, dtype=float)
        self.s_k = None
        self.use_momentum = use_momentum
        self.momentum_func = momentum_func
        self.momentum = None

    def fw_proj(self):
        if self.set == 'l1_ball':
            i_max = np.argmax(np.abs(self.grad))
            self.s_k = np.zeros(self.d, dtype=float)
            self.s_k[i_max] = -1. * float(np.sign(self.grad[i_max]))
        elif self.set == 'l2_ball':
            self.s_k = - self.grad / np.linalg.norm(self.grad, ord=2)
        elif self.set == 'simplex':
            i_min = np.argmin(self.grad)
            self.s_k = np.zeros(self.d, dtype=float)
            self.s_k[i_min] = 1.
        else:
            raise ValueError("Wrong set!")

    def get_next(self):
        learning_rate = self.step(self.k)
        if self.use_momentum:
            self.momentum = self.momentum_func(self.k)
        self.grad, iters_per_step, self.i, self.e_i = self.get_grad(self.x, self.function, self.gamma,
                                                                    self.d, self.k, self.norma)
        self.oracle_counter += iters_per_step
        if self.use_jaguar is True:
            self.h_next = self.h + (self.grad[self.i] - self.h[self.i]) * self.e_i
            self.h = np.copy(self.h_next)
            if self.use_momentum:
                self.y_next = (1 - self.momentum) * self.y + self.momentum * self.h_next + \
                              (1 - self.momentum) * (self.h_next - self.h)
                self.y = np.copy(self.y_next)
                self.grad = np.copy(self.y_next)
            else:
                self.grad = np.copy(self.h_next)

        self.fw_proj()

        return self.x + learning_rate * (self.s_k - self.x)


def projection_simplex(x):
    x_sort = sorted(x, reverse=True)
    rho = 0
    summa = x_sort[0]
    summa_ans = x_sort[0]

    for i in range(1, len(x_sort)):
        summa += x_sort[i]
        if x_sort[i] + 1 / (i + 1) * (1 - summa) > 0:
            rho = i
            summa_ans = summa

    lamb = 1 / (rho + 1) * (1 - summa_ans)
    x_next = np.zeros(len(x_sort))

    for i in range(len(x_next)):
        x_next[i] = max(x[i] + lamb, 0)

    return x_next


def projection_l2(x):
    x_norm = np.linalg.norm(x, ord=2)
    return x / np.linalg.norm(x, ord=2) if x_norm > 1 else x


def projection_l1(x):
    d = len(x)
    y = [0] * d

    def g(llambda, xi):
        summa = 0
        for j in range(d):
            summa += max(np.abs(xi[j]) - llambda, 0)
        return summa - 1

    if norm(x, ord=1) <= 1:
        y = x
    else:
        x_sort = np.sort(np.abs(x))
        m = d - 1
        for i, lmbda in enumerate(x_sort):
            if g(lmbda, x) < 0:
                m = i
                break
        lmbda = 1. / (d - m) * (np.sum(x_sort[m:]) - 1)
        for i in range(d):
            y[i] = np.sign(x[i]) * max(np.abs(x[i]) - lmbda, 0)
        y = np.array(y)
    return y


def get_grad_tpf_jaguar(x, func, gamma_func, d, k, norma):
    gamma = gamma_func(k)
    nabla_f = np.zeros(d, dtype=float)
    i = np.random.randint(d)
    e = np.zeros(d, dtype=float)
    e[i] = 1
    nabla_f += (func(x + gamma * e) - func(x - gamma * e)) / (2. * gamma) * e
    return nabla_f, 2, i, e


def get_grad_tpf_full(x, func, gamma_func, d, k, norma):
    gamma = gamma_func(k)
    nabla_f = np.zeros(d, dtype=float)
    for i in range(d):
        e = np.zeros(d, dtype=float)
        e[i] = 1
        nabla_f += (func(x + gamma * e) - func(x - gamma * e)) / (2. * gamma) * e
    return nabla_f, 2 * d, None, None


def get_grad_tpf(x, func, gamma_func, d, k, norma):
    gamma = gamma_func(k)
    nabla_f = np.zeros(d, dtype=float)
    e = 2 * np.random.rand(d) - 1
    e = e / np.linalg.norm(e, ord=norma)
    nabla_f += (func(x + gamma * e) - func(x - gamma * e)) / (2. * gamma) * e
    return nabla_f, 2, None, e


def get_grad_opf(x, func, gamma_func, d, k, norma):
    gamma = gamma_func(k)
    nabla_f = np.zeros(d, dtype=float)
    e = 2 * np.random.random_sample(d) - 1
    e = e / np.linalg.norm(e, ord=norma)
    nabla_f += d * func(x + gamma * e) * e / gamma
    return nabla_f, 1, None, e


def get_grad_opf_jaguar(x, func, gamma_func, d, k, norma):
    gamma = gamma_func(k)
    nabla_f = np.zeros(d, dtype=float)
    i = np.random.randint(d)
    e = np.zeros(d, dtype=float)
    e[i] = np.random.choice([1, -1], 1)
    nabla_f += func(x + gamma * e) * e / gamma
    return nabla_f, 1, i, e


# метод, генерирующий симметричную матрицу A dxd с собственными значениями из [mu; L]
def generate_matrix(d, mu, L):
    diag = (L - mu) * np.random.random_sample(d) + mu
    sigma = np.diag(diag)
    sigma[0][0] = L
    sigma[d - 1][d - 1] = mu
    rand_matrix = np.random.rand(d, d)
    rand_ort, _, _ = np.linalg.svd(rand_matrix)
    matrix = rand_ort.T @ sigma @ rand_ort
    return matrix


def make_err_plot(iterations_list, errors_list, labels, title, x_label="Iteration number",
                  y_label="The value of the criterion", markers=None, markersize=7, name=None,
                  colors=None, linestyles=None):
    if markers is None:
        markers = [None] * 100
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple',
                  'cyan', 'black', 'olive', 'pink', 'brown']
    if linestyles is None:
        linestyles = ['solid'] * 100

    plt.figure(figsize=(12, 8))
    plt.title(title + "\n logarithmic scale on the y axis", fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)

    for iterations, errors, label, color, marker, linestyle in \
            zip(iterations_list, errors_list, labels, colors, markers, linestyles):
        plt.semilogy(iterations, errors, color=color, label=label, linewidth=2,
                     marker=marker, markersize=markersize, linestyle=linestyle)

    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if name is not None:
        plt.savefig(f"figures/{name}.png")
    plt.show()
