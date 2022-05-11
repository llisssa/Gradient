import numpy as np
import matplotlib.pyplot as plt
import math
import time

def const_step_method(A, b, x, e, imax, alpha):
    """
    Функция с постоянным шагом альфа (alpha)

    Parameters:
    alpha - альфа, переменная вещественного типа, задается с клавиатуры, это сам шаг, который приближает полученные х к истинному значению числа х
    A - матрица коэффициентов
    b - столбец свободных членов
    x - столбец решений
    e - погрешность
    imax - максимальное число итераций


    Returns:
    Сначала находится приближенное значение столбца x, далее в цикле пересчитывается невязка.
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства и количество шагов.
    i - int (целочисленный тип данных), переменная, которая хранит значения следующего объекта
    d - float (вещественный тип данных), квадрат нормы невязки
    x - столбец решений

    residuum - столбец, разность между столбцом b и произведением матрицы А на столбец х, величина ошибки (расхождения) приближенного равенства (невязка)
    d0 - float (вещественный тип данных), значение d

    """
    residuum = b - A.dot(x)
    d = np.transpose(residuum).dot(residuum)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        x = x + (alpha * residuum)
        residuum = b - A.dot(x)
        d = np.transpose(residuum).dot(residuum)
        i += 1
    return x, np.sqrt(d), i

def var_step_method(A, b, x, e, imax):
    """
    Функция с переменным шагом альфа (а), он изменяется в зависимости от величины расхождения
    В данном случае умножаем матрицу коэффициентов на невязку, считаем альфу по формуле.
    Потом находится приближенное значение столбца x, далее в цикле пересчитывается невязка.
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства и количество шагов.

    """
    residuum = b - A.dot(x)
    d = np.transpose(residuum).dot(residuum)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        q = A.dot(residuum)
        alpha = d / (np.transpose(residuum).dot(q))
        x = x + alpha * residuum
        residuum = b - A.dot(x)
        d = np.transpose(residuum).dot(residuum)
        i += 1
    return x, np.sqrt(d), i

def conjugate_gradients_method(A, b, x, e, imax):
    arr = np.empty((0, 2), float)
    i = 0
    residuum = b - A.dot(x)
    d = residuum
    delt = np.transpose(residuum).dot(residuum)
    delt0 = delt
    while (i < imax) and (delt > e**2 * delt0):
        q = A.dot(d)
        alpha = delt / (np.transpose(d).dot(q))
        x = x + alpha * d
        residuum = b - A.dot(x)
        delta = delt
        delt = np.transpose(residuum).dot(residuum)
        betta = delt / delta
        d = residuum + betta * d
        i += 1
        arr = np.append(arr, np.array([x]), axis=0)
    return x, np.sqrt(delt), i, arr

def main():

    print("Matrix 2*2")
    A = np.array([[5, 0], [2, 1]], dtype=float)
    b = np.transpose(np.array([3, 4], dtype=float))
    x = np.transpose(np.array([0, 1], dtype=float))  # [0.6, 2.8]
    e = 0.1
    imax = 100
    alpha = 0.33
    print("Constant step method")
    print(const_step_method(A, b, x, e, imax, alpha))
    print("Variable step method")
    print(var_step_method(A, b, x, e, imax))

    alpha = 0
    alph_arr = np.arange(0, 1, 0.05)
    det = np.zeros(len(alph_arr))
    k = 0
    while alpha < 1:
        x, d, i = const_step_method(A, b, x, e, imax, alpha)
        alpha += 0.05
        det[k] = d
        k += 1
    alpha = 0.33
    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.vlines(alpha, 0, 1, color='r')
    ax.plot(alph_arr, det)
    plt.show()


    print("Matrix 3*3")
    A = np.array([[1, 1, 1], [2, 3, 0], [1, 2, 1]], dtype=float)
    b = np.transpose(np.array([2, 5, -1], dtype=float))
    x = np.transpose(np.array([6, -2, -1], dtype=float))  # [7, -3, -2]
    e = 0.01
    imax = 100
    alpha = 0.28
    print("Constant step method")
    print(const_step_method(A, b, x, e, imax, alpha))
    print("Variable step method")
    print(var_step_method(A, b, x, e, imax))


    alpha = 0
    alph_arr = np.arange(0, 1, 0.05)
    det = np.zeros(len(alph_arr))
    k = 0
    while alpha < 1:
        x, d, i = const_step_method(A, b, x, e, imax, alpha)
        alpha += 0.05
        det[k] = d
        k += 1
    alpha = 0.31
    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.vlines(alpha, 0, 1, color='r')
    ax.plot(alph_arr, det)
    plt.show()


    print("Matrix 4*4")
    # A = np.array([[1, 1, 2, 3], [1, 2, 3, -1], [3, -1, -1, -2], [2, 3, -1, -1]], dtype=float)
    A = np.eye(4) + np.diag(np.ones(3), k=1) + np.diag(np.ones(3), k=-1)
    b = np.transpose(np.array([3, 2, 7, 1], dtype=float))
    x = np.transpose(np.array([0, 4, 0, 3], dtype=float))  # [1, 3, 1, 4]
    e = 0.01
    imax = 1000
    alpha = 0.0001
    print("Constant step method")
    print(const_step_method(A, b, x, e, imax, alpha))
    print("Variable step method")
    print(var_step_method(A, b, x, e, imax))

    alpha = 0
    alph_arr = np.arange(0, 1, 0.05)
    det = np.zeros(len(alph_arr))
    k = 0
    while alpha < 1:
        x, d, i = const_step_method(A, b, x, e, imax, alpha)
        alpha += 0.05
        det[k] = d
        k += 1
    alpha = 0.13
    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.vlines(alpha, 0, 1, color='r')
    ax.plot(alph_arr, det)
    plt.show()

main()