import time
import numpy as np
import random
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pylab as plt


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

        return x - learning_rate * self.gradient(x, self.args)
    
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
            
    
    def projection(self, x):
        
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
        for k in progress(range(self.args['max_steps'])):
            
            self.args['k'] = k + 1
            
            # шаг градиентного спуска
            x_next = self.get_next(x, x_previous, y, k)
            
            if self.args['proj'] is True:
                x_next = self.projection(x_next)
            
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

        errors = np.array(errors)
        times = np.array(times)
        return x, errors, times
    
class MDOptimizer(GDOptimizer):
    
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)
        
    def get_next(self, x, x_previous, y, k):
        
        '''
        Зеркальный метод для симплекса
        '''
        
        learning_rate = self.step(k, self.function, self.gradient, x, self.args)
        sigma = 0
        
        for i, x_i in enumerate(x):
            sigma += np.exp(-learning_rate * self.gradient(x, self.args)[i]) * x_i

        x_next = np.copy(x)
        for i, x_i in enumerate(x):
            x_next[i] = (x_i / sigma) * np.exp(-learning_rate * self.gradient(x, self.args)[i])
        
        return x_next / np.linalg.norm(x_next, ord = 1)
    
class FWOptimizer(GDOptimizer):
    
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)
        
    def get_next(self, x, x_previous, y, k):
        
        '''
        Метод Франка-Вульфа для симплекса
        '''
        
        learning_rate = self.step(k, self.function, self.gradient, x, self.args)

        i_min = np.argmin(self.gradient(x, self.args))
        s_k = np.zeros(len(x), dtype=float)
        s_k[i_min] = 1.

        x_next = x + learning_rate * (s_k - x)

        return x_next
    
# class ProjOptimizer(GDOptimizer):
    
#     def __init__(self, function, gradient, x_0, step, args):
#         GDOptimizer.__init__(self, function, gradient, x_0, step, args)    
    
#     def get_next(self, x, x_previous, y, k):
        
#         """
#         Проекция на симплекс
#         """
        
#         x_sort = sorted(x, reverse=True)
#         rho = 0
#         summa = x_sort[0]
#         summa_ans = x_sort[0]
        
#         for i in range(1, len(x_sort)):
#             summa += x_sort[i]
#             if x_sort[i] + 1 / (i + 1) * (1 - summa) > 0:
#                 rho = i
#                 summa_ans = summa
                
#         lamb = 1 / (rho + 1) * (1 - summa_ans)
#         x_next = np.zeros(len(x_sort))
        
#         for i in range(len(x_next)):
#             x_next[i] = max(x[i] + lamb, 0)
            
#         return x_next
    
class MBFWOptimizer(GDOptimizer):
    
    def __init__(self, function, gradient, x_0, step, args):
        GDOptimizer.__init__(self, function, gradient, x_0, step, args)   
        
    def get_next(self, x, x_previous, y, k):  
        
        """
        Momentum-Based Frank-Wolfe
        """
        
        learning_rate = self.step(k, self.function, self.gradient, x, self.args)

        momentum = self.args['momentum_k'](k, self.function, self.gradient, x, self.args)

        y_k = (1 - momentum) * y + momentum * self.gradient(x, self.args) + \
                 (1 - momentum) * (self.gradient(x, self.args) - self.gradient(x_previous, self.args))

        i_min = np.argmin(y_k)
        z_k = np.zeros(len(y_k), dtype=float)
        z_k[i_min] = 1.

        x_next = x + self.args['gamma'](self.args['k']) * (z_k - x)

        return x_next
    
def get_grad_tpf_v2(x, args):
    
    np.random.seed(args['seed'])
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

def get_grad_tpf_v3(x, args):
    
    np.random.seed(args['seed'])
    func = args['func']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = 1
        
        nabla_f += (func(x + gamma * e, args) -\
                    func(x - gamma * e, args)) / (2. * gamma) * e

    return nabla_f


def get_grad_tpf(x, args):
    
    np.random.seed(args['seed'])
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


def get_grad_opf(x, args):
    
    np.random.seed(args['seed'])
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

def get_grad_opf_v3(x, args):
    
    np.random.seed(args['seed'])
    func = args['func']
    gamma = args['gamma'](args['k'])
    norm = args['norm']
    batch_size = args['batch_size']
    d = args['d']
    nabla_f = np.zeros(d, dtype=float)
    idxs = random.sample(range(d), batch_size)
    for i in idxs:
        e = np.zeros(d, dtype=float)
        e[i] = 1
        
        nabla_f += func(x + gamma * e, args) * e / gamma

    return nabla_f



# переписано до сюда

# функция измения accuracy и mse score модели
# def acc_mse_scores(optimizer, X_test, y_test,
#                    max_steps = 400, step = 10, f_act=np.round):
#     """
#     :param gradient_descent: объект класса GradientDesent
#     :param X_test: матрица объект-признак для тестовой выборки
#     :param y_test: тестовые значения целевой переменной
#     :param max_steps: максимальное кол-во шагов метода
#     :param step: через сколько итераций нужно измерять метрики
#     :param f_act: функция активации
#     :return: итерации, время, accuracy и mse score и то, что возвращает градиентный спуск
#     """

#     optimizer.max_steps = step
#     iterations = []
#     times = []
#     acc_score_list = []
#     mse_score_list = []
#     errors = np.array([])
#     w_pred = gradient_descent.x_0

#     start_time = time.time()
#     for iteration in range(0, max_steps + 1, step):
#         time_now = time.time()
#         w_pred, error = gradient_descent.search()
#         gradient_descent.x_0 = w_pred

#         y_pred = f_act(X_test @ w_pred)

#         iterations.append(iteration + step)
#         times.append(time_now - start_time)
#         acc_score_list.append(accuracy_score(y_test, y_pred))
#         mse_score_list.append(mean_squared_error(y_test, y_pred))
#         errors = np.hstack([errors, error[-1]])

#     return iterations, times, acc_score_list, mse_score_list, w_pred, errors


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

# функция для отслеживая процесса
def progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )