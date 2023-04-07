import time
import numpy as np
import random
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pylab as plt
from tqdm import tqdm
from tqdm import trange
import sys

# в args: max steps, eps, criterium, gradient_true, x_true, momentum

class GDOptimizer:
    def __init__(self, function, gradient, x_0, step, args):
        '''
        :param function: целевая функция
        :param gradient: градиант целевой функции
        :param x_0: стартовая точка
        :param step: функция для вычисления шага метода
        :param args: гиперпараметры function, gradient и других функций
        '''

        self.function = function
        self.gradient = gradient
        self.x_0 = x_0
        self.step = step
        self.args = args
    def get_next(self, x, x_previous, y, k):
        '''
        Градиентный спуск
        '''
        learning_rate = self.step(x, k, self.function, self.gradient, self.args)
        #########
        grad = self.args['grad_curr']
        if k % (self.args['d'] - 1) == 0:
            self.args['batch_size'] = self.args['d']
            grad = self.gradient(x, self.args)
        else:
            self.args['batch_size'] = 1
            grad[self.args['i']] = self.gradient(x, self.args)[self.args['i']]
        self.args['grad_curr'] = grad
        ##########
        return x - learning_rate * self.gradient(x, self.args), 1
    
    def get_error(self, function, gradient, args, x, x_previous):
        if self.args['criterium'] == 'x_k - x^*':
            return norm(x - self.args['x_sol'], ord=2)
            
        elif self.args['criterium'] == 'f(x_k) - f(x^*)':
            return self.function(x, self.args) \
                       - self.function(self.args['x_sol'], self.args)
            
        elif self.args['criterium'] == 'x_k+1 - x_k':
            return norm(x - x_previous, ord=2)
            
        elif self.args['criterium'] == 'f(x_k+1) - f(x_k)':
            return np.abs(self.function(x, self.args) - self.function(x_previous, self.args))
            
        elif self.args['criterium'] == 'nabla(f)(x_k)':
            return norm(self.gradient(x, self.args), ord=2)
            
        elif self.args['criterium'] == 'gap':
            return self.args['gradient_true'](x, self.args) @ x \
                        - np.min(self.args['gradient_true'](x, self.args))
        
        else:
            raise ValueError('Wrong criterium!')
            
    def projection_simplex(self, x):
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
            
    

    def projection_l2(self, x):
        """
        Проекция на l2-ball
        """
        x_norm = np.linalg.norm(x, ord = 2)
        if x_norm > 1:
            x_next = x / np.linalg.norm(x, ord = 2)
        else:
            x_next = x
        return x_next
    
    def projection_l1(self, x):
        """
        Проекция на l1-ball
        """
        '''
        if np.linalg.norm(x, ord = 1) <= 1:
            return x
        else:
            arr = []
            left = 1
            delta = 0
            target = 0
            last_i = 0
            for i in range(self.args['d']):
                arr.append([np.absolute(x[i]), i])
            arr = sorted(arr, key = lambda a: abs(a[0]), reverse=True)
            for i in range(self.args['d']):
                if (i != self.args['d'] - 1) and (arr[i][0] - arr[i + 1][0] < left):
                    left = left - (arr[i][0] - arr[i + 1][0])
                else:
                    delta = left / (i + 1)
                    target = arr[i][0] - delta
                    last_i = i
                    break
            for i in range(self.args['d']):
                if i <= last_i:
                    arr[i][0] = (arr[i][0] - target) * np.sign(x[arr[i][1]])
                else:
                    arr[i][0] = 0
            arr = sorted(arr, key = lambda a: a[1])
            return np.asarray([i[0] for i in arr])
            
        '''
        d = len(x)
        y = [0] * d

        def g(lmbda, x):
            sum = 0
            for i in range(d):
                sum += max(np.abs(x[i]) - lmbda, 0)
            return sum - 1

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

    def search(self):
        '''
        :return: x - значение аргумента, на котором достигается минимум функции
        :return: errors - вектор значений критерия на каждой итерации
        :return: times - вектор времен работы
        '''
        x = self.x_0
        x_previous = self.x_0
        y = self.x_0
        theta = 1
        start_time = time.time()
        times = [0]
        errors = []
        for k in trange(self.args['max_steps'], file=sys.stdout, colour="green"):
            # шаг градиентного спуска
            x_next, iteration = self.get_next(x, x_previous, y, k)
            self.args['k'] = k + iteration

            if self.args['use_proj'] is True:
                if self.args['set'] == 'l1_ball':
                    x_next = self.projection_l1(x_next)
                elif self.args['set'] == 'l2_ball':
                    x_next = self.projection_l2(x_next)
                elif self.args['set'] == 'simplex':
                    x_next = self.projection_simplex(x_next)
                else:
                    raise ValueError("Wrong set!")
            
            x_previous = x
            x = x_next
            
            # добавление новой ошибки в вектор errors
            error = self.get_error(self.function, self.gradient, self.args, x, x_previous)
            errors.append(error)
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


class MDOptimizer(GDOptimizer):
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)
        
    def get_next(self, x, x_previous, y, k):
        '''
        Зеркальный метод
        '''
        learning_rate = self.step(k, self.function, self.gradient, x, self.args)
        sigma = 0
        
        for i, x_i in enumerate(x):
            sigma += np.exp(-learning_rate * self.gradient(x, self.args)[i]) * x_i

        x_next = np.copy(x)
        for i, x_i in enumerate(x):
            x_next[i] = (x_i / sigma) * np.exp(-learning_rate * self.gradient(x, self.args)[i])
        
        return x_next / np.linalg.norm(x_next, ord=1), 1


class FWOptimizer(GDOptimizer):
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)
        
    def get_next(self, x, x_previous, y, k):
        '''
        Метод Франка-Вульфа для симплекса
        '''
        learning_rate = self.step(k, self.function, self.gradient, x, self.args)
        
        s_k = None
        #########
        grad = self.args['grad_curr']
        if k % (self.args['d'] - 1) == 0:
            self.args['batch_size'] = self.args['d']
            grad = self.gradient(x, self.args)
        else:
            self.args['batch_size'] = 1
            grad[self.args['i']] = self.gradient(x, self.args)[self.args['i']]
        self.args['grad_curr'] = grad
        ##########
        if self.args['set'] == 'l1_ball':
            i_max = np.argmax(np.abs(grad))
            s_k = np.zeros(len(x), dtype=float)
            s_k[i_max] = -1. * float(np.sign(grad[i_max]))
        elif self.args['set'] == 'l2_ball':
            s_k = - grad / np.linalg.norm(grad, ord = 2)
        elif self.args['set'] == 'simplex':
            i_min = np.argmin(grad)
            s_k = np.zeros(len(x), dtype=float)
            s_k[i_min] = 1.
        else:
            raise ValueError("Wrong set!")

        x_next = x + learning_rate * (s_k - x)

        return x_next, 1


class MBFWOptimizer(GDOptimizer):
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)   
        
    def get_next(self, x, x_previous, y, k):
        """
        Momentum-Based Frank-Wolfe
        """
        learning_rate = self.step(k, self.function, self.gradient, x, self.args)

        momentum = self.args['momentum_k'](k, self.function, self.gradient, x, self.args)
        ######
        grad_prev = self.args['grad_curr']
        if k % (self.args['d'] - 1) == 0:
            self.args['batch_size'] = self.args['d']
            grad_next = self.gradient(x, self.args)
        else:
            self.args['batch_size'] = 1
            grad_next = grad_prev
            grad_next[self.args['i']] = self.gradient(x, self.args)[self.args['i']]
        self.args['grad_curr'] = grad_next
        #######

        y_k = (1 - momentum) * y + momentum * grad_next + \
              (1 - momentum) * (grad_next - grad_prev)

        s_k = None
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
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)

    def get_next(self, x, x_previous, y, k):
        """
        Faster Zeroth-Order Conditional Gradient Method
        """
        def condg(g, u, gamma, eta):
            t = 1
            u_t = u
            s_k = None
            while True:
                grad = g + 1./gamma * (u_t - u)
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

                alpha_t = min(1, ((1./gamma * (u - u_t) - g) @ (s_k - u_t)) / \
                              (1./gamma * (np.linalg.norm(s_k - u_t, ord=2)**2)))
                u_t = (1 - alpha_t) * u_t + alpha_t * s_k
                t += 1

        learning_rate = self.step(k, self.function, self.gradient, x, self.args)
        eta = self.args['eta'](k, self.function, self.gradient, x, self.args)

        x_next, t = condg(self.gradient(x_previous, self.args), x_previous, learning_rate, eta)
        return x_next, 1


def get_grad_tpf_jaguar(x, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = [-1, 1][random.randrange(2)]
        ########
        args['i'] = i
        ########
        nabla_f += (func(x + gamma * e, args) - \
                    func(x - gamma * e, args)) / (2. * gamma) * e

    return (float(d) / float(batch_size)) * nabla_f


def get_grad_tpf_lame_v2(x, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    for i in range(d):
        e = np.zeros(d, dtype=float)
        e[i] = 1
        
        nabla_f += (func(x + gamma * e, args) -\
                    func(x - gamma * e, args)) / (2. * gamma) * e

    return nabla_f


def get_grad_tpf_lame_v1(x, args):
    func = args['func']
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

    return 1. /batch_size * nabla_f


def get_grad_opf_lame(x, args):
    func = args['func']
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

    return 1./batch_size * nabla_f


def get_grad_opf_jaguar(x, args):
    func = args['func']
    gamma = args['gamma'](args['k'])
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = [-1, 1][random.randrange(2)]
        ########
        args['i'] = i
        ########
        nabla_f += func(x + gamma * e, args) * e / gamma

    return nabla_f

# функция для отрисовки графиков сходимости
def make_err_plot(iterations_list, errors_list, labels, title, x_label="Iteration number",
                  y_label="The value of the criterion", markers=["^"]*100, markersize=7):
    """
    :param iterations_list: список из итераций для кадого вектора ошибок
    :param errors_list: список векторов ошибок
    :param labels: заголовок для каждого вектора ошибок
    :param title: заголовок графика
    :param x_label: подпись оси OX
    :param ylabel: подпись оси OY
    :param markers: маркеры на точки
    :param markersize: размер маркеров
    :return: Функкия отрисовывает график ошибки от кол-ва итераций в лог. масштабе
    """

    colors = ['red', 'green', 'blue', 'orange', 'purple',
              'cyan', 'black', 'olive', 'pink', 'brown']

    plt.figure(figsize=(12, 8))
    plt.title(title + "\n logarithmic scale on the y axis", fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)

    for iterations, errors, label, color, marker in \
            zip(iterations_list, errors_list, labels, colors, markers):
        plt.semilogy(iterations, errors, color = color, label = label, linewidth = 2,
                     marker = marker, markersize = markersize)

    #plt.xticks(np.arange(0, 200, 40), fontsize=25)
    #plt.yticks(fontsize=25)
    #plt.ylim(10 ** (-7), 3)
    #plt.xlim(-50, 1050)
    plt.grid()
    plt.legend(fontsize = 15)
    plt.show()

# функция для отрисовки графиков метрик
def make_score_plot(iterations_list, acc_List, mse_List, labels,
                    title_first = "График зависимости accuracy score от номера итерации",
                    title_second = "График зависимости MSE от номера итерации",
                    xlabel = "Номер итерации", ylabel_first = "Accuracy score",
                    ylabel_second = "Mean squared error", markers = ["^"] * 100,
                    two_plots = True):
    """
    :param iterations_list: список из итераций для кадого вектора результатов
    :param acc_List: список списков результатов измерения accuracy_score
    :param mse_List: список списков результатов измерения mean_squared_error
    :param labels: заголовок для каждого вектора результатов
    :param title_first: заголовок 1 графика
    :param title_second: заголовок 2 графика
    :param xlabel: подпись оси ОХ
    :param ylabel_first: подпись оси ОY на 1 графике
    :param ylabel_second: подпись оси ОY на 2 графике
    :param two_plots: нужно ли рисовать 2 графика или нет (True если нужно)
    :return: функция рисует 1 или 2 графика - зависимость accuracy score и mse
    """
    colors = ['red', 'green', 'cyan', 'blue', 'purple',
              'black', 'olive', 'pink', 'brown', 'orange']

    if two_plots:
        _, ax = plt.subplots(1, 2, figsize = (25, 10))
        for iterations, acc_score_list, mse_score_list, \
                label, color, marker in zip(iterations_list, acc_List,
                                            mse_List, labels, colors, markers):
            ax[0].plot(iterations, acc_score_list,
                       color=color, label=label, linewidth=2, marker=marker, markersize=7)
            ax[0].set_title(title_first, fontsize=15)
            ax[0].set_xlabel(xlabel, fontsize=15)
            ax[0].set_ylabel(ylabel_first, fontsize=15)
            ax[1].plot(iterations, mse_score_list,
                       color=color, label=label, linewidth=2, marker=marker, markersize=7)
            ax[1].set_title(title_second, fontsize=15)
            ax[1].set_xlabel(xlabel, fontsize=15)
            ax[1].set_ylabel(ylabel_second, fontsize=15)

        ax[0].grid()
        ax[1].grid()
        ax[0].legend(fontsize=15)
        ax[1].legend(fontsize=15)
    else:
        plt.figure(figsize=(12, 8))
        plt.title(title_first, fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel_first, fontsize=15)

        for iterations, acc_score_list, label, color, marker \
                in zip(iterations_list, acc_List, labels, colors, markers):
            plt.plot(iterations, acc_score_list, color=color, label=label,
                     linewidth=2, marker=marker, markersize=7)
        plt.legend(fontsize=15)
    plt.show()


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