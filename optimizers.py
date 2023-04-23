import time
import numpy as np
import random
from numpy.linalg import norm
from matplotlib import pylab as plt
from tqdm import trange
import sys


class GDOptimizer:
    def __init__(self, args):
        """
        :param args: параметры запуска модели
        """
        self.args = args

    def get_next(self, x, x_previous, y, k):
        """
        Градиентный спуск
        """
        learning_rate = self.args['step'](self.args)
        if self.args['jaguar'] is True:
            grad = self.args['grad_curr']
            if random.random() < self.args['prob']:
                self.args['batch_size'] = self.args['d']
                grad = self.args['gradient'](x, self.args)
            else:
                self.args['batch_size'] = 1
                grad[self.args['i']] = self.args['gradient'](x, self.args)[self.args['i']]
            self.args['grad_curr'] = grad
        elif self.args['sega'] is True:
            if random.random() < self.args['prob']:
                self.args['batch_size'] = self.args['d']
                grad_next = self.args['gradient'](x, self.args)
                self.args['h'] = np.copy(grad_next)
            else:
                self.args['batch_size'] = 1
                e_i = np.zeros(self.args['d'])
                grad = self.args['gradient'](x, self.args)
                i = self.args['i']
                e_i[i] = 1.
                h_next = self.args['h'] + (grad[i] - self.args['h'][i]) * e_i
                grad_next = self.args['d'] * (grad[i] - self.args['h'][i]) * e_i + self.args['h']
                self.args['grad_curr'] = np.copy(grad_next)
                self.args['h'] = np.copy(h_next)
            grad = grad_next
        else:
            grad = self.args['gradient'](x, self.args)

        return x - learning_rate * grad, 1

    def get_error(self, x, x_previous):
        if self.args['criterium'] == 'x_k - x^*':
            return norm(x - self.args['x_sol'], ord=2)

        elif self.args['criterium'] == 'f(x_k) - f(x^*)':
            return self.args['function'](x, self.args) \
                - self.args['function'](self.args['x_sol'], self.args)

        elif self.args['criterium'] == 'x_k+1 - x_k':
            return norm(x - x_previous, ord=2)

        elif self.args['criterium'] == 'f(x_k+1) - f(x_k)':
            return np.abs(self.args['function'](x, self.args) - self.args['function'](x_previous, self.args))

        elif self.args['criterium'] == 'nabla(f)(x_k)':
            return norm(self.args['gradient'](x, self.args), ord=2)

        elif self.args['criterium'] == 'gap':
            return self.args['gradient_true'](x, self.args) @ x \
                - np.min(self.args['gradient_true'](x, self.args))

        else:
            raise ValueError('Wrong criterium!')

    def search(self):
        """
        :return: x - значение аргумента, на котором достигается минимум функции
        :return: errors - вектор значений критерия на каждой итерации
        :return: times - вектор времен работы
        """
        x = self.args['x_0']
        x_previous = self.args['x_0']
        y = self.args['x_0']
        start_time = time.time()
        times = [0]
        errors = []
        for k in trange(self.args['max_steps'], file=sys.stdout, colour="green"):
            # шаг градиентного спуска
            x_next, iteration = self.get_next(x, x_previous, y, k)
            self.args['k'] = k + iteration

            if self.args['use_proj'] is True:
                if self.args['set'] == 'l1_ball':
                    x_next = projection_l1(x_next)
                elif self.args['set'] == 'l2_ball':
                    x_next = projection_l2(x_next)
                elif self.args['set'] == 'simplex':
                    x_next = projection_simplex(x_next)
                else:
                    raise ValueError("Wrong set!")

            x_previous = x
            x = x_next

            # добавление новой ошибки в вектор errors
            error = self.get_error(x, x_previous)
            errors.append(error)
            self.args['oracle_calls'].append(self.args['oracle_counter'])
            time_now = time.time()
            times.append(time_now - start_time)

            # критерий остановки
            if error <= self.args['eps']:
                break
            if self.args['k'] > self.args['max_steps']:
                break

        errors = np.array(errors)
        times = np.array(times)
        return x, errors, times


class DGDOptimizer(GDOptimizer):
    def __init__(self, args):
        GDOptimizer.__init__(self, args)

    def get_next(self, x, x_previous, y, k):
        """
        Распредленный градиентынй спуск
        """
        learning_rate = self.args['step'](self.args)

        idxs = random.sample(range(self.args['d']), self.args['d'] * self.args['percent'] // 100)
        Q_grad_f = np.zeros(self.args['d'])

        for j in idxs:
            Q_grad_f[j] = self.args['gradient'][j]

        return x - learning_rate * Q_grad_f


class MDOptimizer(GDOptimizer):
    def __init__(self, args):
        GDOptimizer.__init__(self, args)

    def get_next(self, x, x_previous, y, k):
        """
        Зеркальный метод
        """
        learning_rate = self.args['step'](self.args)
        sigma = 0

        for i, x_i in enumerate(x):
            sigma += np.exp(-learning_rate * self.args['gradient'](x, self.args)[i]) * x_i

        x_next = np.copy(x)
        for i, x_i in enumerate(x):
            x_next[i] = (x_i / sigma) * np.exp(-learning_rate * self.args['gradient'](x, self.args)[i])

        return x_next / np.linalg.norm(x_next, ord=1), 1


class FWOptimizer(GDOptimizer):
    def __init__(self, args):
        GDOptimizer.__init__(self, args)

    def get_next(self, x, x_previous, y, k):
        """
        Метод Франка-Вульфа для симплекса
        """
        learning_rate = self.args['step'](self.args)

        if self.args['jaguar'] is True:
            #########
            grad = self.args['grad_curr']
            if random.random() < self.args['prob']:
                self.args['batch_size'] = self.args['d']
                grad = self.args['gradient'](x, self.args)
            else:
                self.args['batch_size'] = 1
                grad[self.args['i']] = self.args['gradient'](x, self.args)[self.args['i']]
            self.args['grad_curr'] = grad
            ##########
        elif self.args['sega'] is True:
            if random.random() < self.args['prob']:
                self.args['batch_size'] = self.args['d']
                grad_next = self.args['gradient'](x, self.args)
                self.args['h'] = np.copy(grad_next)
            else:
                self.args['batch_size'] = 1
                e_i = np.zeros(self.args['d'])
                grad = self.args['gradient'](x, self.args)
                i = self.args['i']
                e_i[i] = 1.
                h_next = self.args['h'] + (grad[i] - self.args['h'][i]) * e_i
                grad_next = self.args['d'] * (grad[i] - self.args['h'][i]) * e_i + self.args['h']
                self.args['grad_curr'] = np.copy(grad_next)
                self.args['h'] = np.copy(h_next)
            grad = grad_next
        else:
            grad = self.args['gradient'](x, self.args)

        if self.args['set'] == 'l1_ball':
            i_max = np.argmax(np.abs(grad))
            s_k = np.zeros(len(x), dtype=float)
            s_k[i_max] = -1. * float(np.sign(grad[i_max]))
        elif self.args['set'] == 'l2_ball':
            s_k = - grad / np.linalg.norm(grad, ord=2)
        elif self.args['set'] == 'simplex':
            i_min = np.argmin(grad)
            s_k = np.zeros(len(x), dtype=float)
            s_k[i_min] = 1.
        else:
            raise ValueError("Wrong set!")

        x_next = x + learning_rate * (s_k - x)

        return x_next, 1


class MBFWOptimizer(GDOptimizer):
    def __init__(self, args):
        GDOptimizer.__init__(self, args)

    def get_next(self, x, x_previous, y, k):
        """
        Momentum-Based Frank-Wolfe
        """
        learning_rate = self.args['step'](self.args)

        momentum = self.args['momentum_k'](self.args)

        if self.args['jaguar'] is True:
            #########
            grad = self.args['grad_curr']
            grad_prev = self.args['grad_curr']
            if random.random() < self.args['prob']:
                self.args['batch_size'] = self.args['d']
                grad = self.args['gradient'](x, self.args)
            else:
                self.args['batch_size'] = 1
                grad[self.args['i']] = self.args['gradient'](x, self.args)[self.args['i']]
            self.args['grad_curr'] = grad
            ##########
            grad_next = grad
        elif self.args['sega'] is True:
            if random.random() < self.args['prob']:
                grad_prev = np.copy(self.args['grad_curr'])
                self.args['batch_size'] = self.args['d']
                grad_next = self.args['gradient'](x, self.args)
                self.args['h'] = np.copy(grad_next)
            else:
                self.args['batch_size'] = 1
                grad_prev = np.copy(self.args['grad_curr'])
                e_i = np.zeros(self.args['d'])
                grad = self.args['gradient'](x, self.args)
                i = self.args['i']
                e_i[i] = 1.
                h_next = self.args['h'] + (grad[i] - self.args['h'][i]) * e_i
                grad_next = self.args['d'] * (grad[i] - self.args['h'][i]) * e_i + self.args['h']
                self.args['grad_curr'] = np.copy(grad_next)
                self.args['h'] = np.copy(h_next)
        else:
            grad_next = self.args['gradient'](x, self.args)
            grad_prev = self.args['gradient'](x_previous, self.args)

        y_k = (1 - momentum) * y + momentum * grad_next + \
              (1 - momentum) * (grad_next - grad_prev)

        grad = y_k
        if self.args['set'] == 'l1_ball':
            i_max = np.argmax(np.abs(grad))
            s_k = np.zeros(len(x), dtype=float)
            s_k[i_max] = -1. * np.sign(grad[i_max])
        elif self.args['set'] == 'l2_ball':
            s_k = - grad / np.linalg.norm(grad, ord=2)
        elif self.args['set'] == 'simplex':
            i_min = np.argmin(grad)
            s_k = np.zeros(len(x), dtype=float)
            s_k[i_min] = 1.
        else:
            raise ValueError("Wrong set!")

        x_next = x + learning_rate * (s_k - x)

        return x_next, 1


class FZCGSOptimizer(GDOptimizer):
    def __init__(self, args):
        GDOptimizer.__init__(self, args)

    def get_next(self, x, x_previous, y, k):
        """
        Faster Zeroth-Order Conditional Gradient Method
        """

        def condg(g, u, gamma, eta):
            t = 1
            u_t = u
            while True:
                grad = g + 1. / gamma * (u_t - u)
                if self.args['set'] == 'l1_ball':
                    i_max = np.argmax(np.abs(grad))
                    s_k = np.zeros(len(grad), dtype=float)
                    s_k[i_max] = -1. * np.sign(grad[i_max])
                elif self.args['set'] == 'l2_ball':
                    s_k = - grad / np.linalg.norm(grad, ord=2)
                elif self.args['set'] == 'simplex':
                    i_min = np.argmin(grad)
                    s_k = np.zeros(len(grad), dtype=float)
                    s_k[i_min] = 1.
                else:
                    raise ValueError("Wrong set!")

                v = grad @ (u_t - s_k)

                if v <= eta:
                    return u_t, t

                alpha_t = min(1, ((1. / gamma * (u - u_t) - g) @ (s_k - u_t)) /
                              (1. / gamma * (np.linalg.norm(s_k - u_t, ord=2) ** 2)))
                u_t = (1 - alpha_t) * u_t + alpha_t * s_k
                t += 1

        learning_rate = self.args['step'](self.args)
        eta = self.args['eta'](k, self.args['function'], self.args['gradient'], x, self.args)
        if self.args['jaguar'] is True:
            #########
            grad = self.args['grad_curr']
            if random.random() < self.args['prob']:
                self.args['batch_size'] = self.args['d']
                grad = self.args['gradient'](x, self.args)
            else:
                self.args['batch_size'] = 1
                grad[self.args['i']] = self.args['gradient'](x, self.args)[self.args['i']]
            self.args['grad_curr'] = grad
            ##########
        else:
            grad = self.args['gradient'](x, self.args)

        x_next, t = condg(grad, x_previous, learning_rate, eta)
        return x_next, 1


def projection_simplex(x):
    """
    Проекция на симплекс
    """
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
    """
    Проекция на l2-ball
    """
    x_norm = np.linalg.norm(x, ord=2)
    if x_norm > 1:
        x_next = x / np.linalg.norm(x, ord=2)
    else:
        x_next = x
    return x_next


def projection_l1(x):
    """
    Проекция на l1-ball
    """
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


def get_grad_tpf_jaguar(x, args):
    func = args['function']
    gamma = args['gamma'](args['k'])
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = [-1, 1][random.randrange(2)]
        args['i'] = i
        nabla_f += (func(x + gamma * e, args) -
                    func(x - gamma * e, args)) / (2. * gamma) * e
        args['oracle_counter'] += 2

    return (float(d) / float(batch_size)) * nabla_f


def get_grad_tpf_lame_v2(x, args):
    func = args['function']
    gamma = args['gamma'](args['k'])
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    for i in range(d):
        e = np.zeros(d, dtype=float)
        e[i] = 1
        nabla_f += (func(x + gamma * e, args) -
                    func(x - gamma * e, args)) / (2. * gamma) * e
        args['oracle_counter'] += 2

    return nabla_f


def get_grad_tpf_lame_v1(x, args):
    func = args['function']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)

    for i in range(batch_size):
        np.random.seed(random.randint(1, 10000))
        e = 2 * np.random.rand(d) - 1
        e = e / np.linalg.norm(e, ord=norm)
        nabla_f += (func(x + gamma * e, args) - func(x - gamma * e, args)) / (2. * gamma) * e
        args['oracle_counter'] += 2

    return 1. / batch_size * nabla_f


def get_grad_opf_lame(x, args):
    func = args['function']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)

    for _ in range(batch_size):
        np.random.seed(random.randint(1, 10000))
        e = 2 * np.random.random_sample(d) - 1
        e = e / np.linalg.norm(e, ord=norm)
        nabla_f += d * func(x + gamma * e, args) * e / gamma
        args['oracle_counter'] += 1

    return 1. / batch_size * nabla_f


def get_grad_opf_jaguar(x, args):
    func = args['function']
    gamma = args['gamma'](args['k'])
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = [-1, 1][random.randrange(2)]
        args['i'] = i
        nabla_f += func(x + gamma * e, args) * e / gamma
        args['oracle_counter'] += 1

    return nabla_f


# метод, генерирующий симметричную матрицу A dxd с собственными значениями из [mu; L]
def generate_A(d, mu, L, seed):
    """
    :param: d - размерность матрицы A
    :param: mu - минимальное собственное число A
    :param: L - максимальное собственное число A
    :param: seed - параметр для фиксации эксперимента

    :return: Симметричная матрица A dxd с собственными значениями из [mu; L]
    """
    np.random.seed(seed)

    # Генерация диагональной матрицы Sigma
    diag = (L - mu) * np.random.random_sample(d) + mu
    sigma = np.diag(diag)
    sigma[0][0] = L
    sigma[d - 1][d - 1] = mu

    # Генерация случайной матрицы U
    rand_matrix = np.random.rand(d, d)
    rand_ort, _, _ = np.linalg.svd(rand_matrix)

    # Получение нашей матрицы A
    A = rand_ort.T @ sigma @ rand_ort

    return A


# функция для отрисовки графиков сходимости
def make_err_plot(iterations_list, errors_list, labels, title, x_label="Iteration number",
                  y_label="The value of the criterion", markers=None, markersize=7, name=None,
                  colors=None, linestyles=None):
    """
    :param iterations_list: список из итераций для кадого вектора ошибок
    :param errors_list: список векторов ошибок
    :param labels: заголовок для каждого вектора ошибок
    :param title: заголовок графика
    :param x_label: подпись оси OX
    :param y_label: подпись оси OY
    :param markers: маркеры на точки
    :param markersize: размер маркеров
    :param name: названия сохраненного файла с графиком
    :param colors: массив цветов
    :param linestyles: стили линий
    :return: Функкия отрисовывает график ошибки от кол-ва итераций в лог. масштабе
    """

    if markers is None:
        markers = ["^"] * 100
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
