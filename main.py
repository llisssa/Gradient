import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as ln

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
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства и количество итераций.
    i - int (целочисленный тип данных), переменная, которая хранит значения следующего объекта
    d - float (вещественный тип данных), квадрат нормы невязки
    x - столбец решений

    residual - столбец, разность между столбцом b и произведением матрицы А на столбец х, величина ошибки (расхождения) приближенного равенства (невязка)
    d0 - float (вещественный тип данных), значение d

    """
    residual = b - A.dot(x)
    d = np.transpose(residual).dot(residual)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        x = x + (alpha * residual)
        residual = b - A.dot(x)
        d = np.transpose(residual).dot(residual)
        i += 1
    return x, np.sqrt(d), i

def var_step_method(A, b, x, e, imax):
    """
    Функция с переменным шагом альфа (аlpha), он изменяется в зависимости от величины расхождения
    В данном случае умножаем матрицу коэффициентов на невязку, считаем альфу по формуле.
    Потом находится приближенное значение столбца x, далее в цикле пересчитывается невязка.
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства и количество шагов.

    """
    i = 0
    residual = b - A.dot(x)
    d = np.transpose(residual).dot(residual)
    d0 = d
    while (i < imax) and (d > e**2 * d0):
        q = A.dot(residual)
        alpha = d / (np.transpose(residual).dot(q))
        x = x + alpha * residual
        residual = b - A.dot(x)
        d = np.transpose(residual).dot(residual)
        i += 1
    return x, np.sqrt(d), i

def var_step_method_visualisation(A, b, x, e, imax):
    """
    Функция с переменным шагом альфа (аlpha), он изменяется в зависимости от величины расхождения
    В данном случае умножаем матрицу коэффициентов на невязку, считаем альфу по формуле.
    Потом находится приближенное значение столбца x, далее в цикле пересчитывается невязка.
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства и количество шагов.

    """
    arr = np.empty((0, 2), float)
    residual = b - A.dot(x)
    d = np.transpose(residual).dot(residual)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        q = A.dot(residual)
        alpha = d / (np.transpose(residual).dot(q))
        x = x + alpha * residual
        residual = b - A.dot(x)
        d = np.transpose(residual).dot(residual)
        i += 1
        arr = np.append(arr, np.array([x]), axis=0)
    return x, np.sqrt(d), i, arr

def conjugate_gradients_method(A, b, x, e, imax):
    """
    Функция метода сопряженных градиентов.
    Так же как и в предыдущих случаях, умножаем матрицу коэффициентов на невязку, считаем альфу по формуле.
    Находится приближенное значение столбца x, далее в цикле пересчитывается невязка.

    Returns:
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства, количество итераций и массив из х.
    i - int (целочисленный тип данных), переменная, которая хранит значения следующего объекта
    delt - float (вещественный тип данных), квадрат нормы невязки
    x - столбец решений

    """
    arr = np.empty((0, 2), float)
    i = 0
    residual = b - A.dot(x)
    d = residual
    delt = np.transpose(residual).dot(residual)
    delt0 = delt
    while (i < imax) and (delt > e**2 * delt0):
        q = A.dot(d)
        alpha = delt / (np.transpose(d).dot(q))
        x = x + alpha * d
        residual = b - A.dot(x)
        delta = delt
        delt = np.transpose(residual).dot(residual)
        betta = delt / delta
        d = residual + betta * d
        i += 1
        arr = np.append(arr, np.array([x]), axis=0)
    return x, np.sqrt(delt), i, arr


def conjugate_gradients_Newton_Raphson(A, b, x, imax, e, jmax, epsilon):
    i = 0
    k = 0
    x_arr = np.empty((0, 2), float)
    residual = -1 * (A.dot(x) - b)
    d = residual
    delta_new = np.transpose(residual).dot(residual)
    delta0 = delta_new
    n = 50
    m = A
    while (i < imax) and (delta_new > e**2 * delta0):
        j = 0
        delta_d = np.transpose(d).dot(d)
        alpha = -1 * (np.transpose(A.dot(x) - b).dot(d)) / (np.transpose(d).dot(m).dot(d))
        x = x + alpha * d
        j += 1
        while (j < jmax) and (alpha ** 2 * delta_d > epsilon ** 2):
            alpha = -1 * (np.transpose(A.dot(x) - b).dot(d)) / (np.transpose(d).dot(m).dot(d))
            x = x + alpha * d
            x_arr = np.append(x_arr, np.array([x]), axis=0)
            j += 1
        residual = -1 * (A.dot(x) - b)
        delta_old = delta_new
        delta_new = np.transpose(residual).dot(residual)
        betta = delta_new / delta_old
        d = residual + betta * d
        k += 1
        if k == n or np.transpose(residual).dot(d) <= 0:
            d = residual
            k = 0
        i += 1
    return x, np.sqrt(delta_new), i, x_arr

def steepest_decent():

    print("Matrix 2*2")
    A = np.array([[2, -1], [-1, 3]], dtype=float)
    b = np.transpose(np.array([3, 4], dtype=float))
    x = np.transpose(np.array([1, 1], dtype=float))  # [2.6, 2.2]
    e = 0.1
    imax = 100
    alpha = 0.3
    print("Constant step method")
    print(const_step_method(A, b, x, e, imax, alpha))
    print("Variable step method")
    print(var_step_method(A, b, x, e, imax))


    print("Matrix 3*3")
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
    b = np.transpose(np.array([1, 2, 3], dtype=float))
    x = np.transpose(np.array([1, 2, 2], dtype=float))  # [2.5, 4, 3.5]
    e = 0.01
    imax = 100
    alpha = 0.3
    print("Constant step method")
    print(const_step_method(A, b, x, e, imax, alpha))
    print("Variable step method")
    print(var_step_method(A, b, x, e, imax))


    print("Matrix 4*4")
    A = np.eye(4) + np.diag(np.ones(3), k=1) + np.diag(np.ones(2), k=-2) + np.diag(np.ones(2), k=2) + np.diag(
        np.ones(1), k=3) + np.diag(np.ones(1), k=-3)
    b = np.transpose(np.array([3, 3, 3, 5], dtype=float))
    x = np.transpose(np.array([4, -1, 4, -1], dtype=float))  # [5, 0, 5, -2]
    e = 0.01
    imax = 50
    alpha = 0.3
    print("Constant step method")
    print(const_step_method(A, b, x, e, imax, alpha))
    print("Variable step method")
    print(var_step_method(A, b, x, e, imax))

def graphs_steepest_decent():
    A = np.array([[2, -1], [-1, 3]], dtype=float)
    b = np.transpose(np.array([3, 4], dtype=float))
    x = np.transpose(np.array([1, 1], dtype=float))
    e = 0.1
    imax = 100
    alph_arr = np.arange(0, 1, 0.05)
    det = np.zeros(len(alph_arr))
    k = 0
    for alpha in alph_arr:
        x, d, i = const_step_method(A, b, x, e, imax, alpha)
        det[k] = d
        k += 1
    alpha_f = np.transpose(b - A.dot(x)).dot(b - A.dot(x)) / (np.transpose(b - A.dot(x)).dot(A.dot(b - A.dot(x))))
    fig, ax = plt.subplots()
    plt.title("Graph for matrix 2*2")
    plt.xlabel("Alpha")
    plt.ylabel("Residual")
    plt.yscale('log')
    ax.plot(alph_arr, det, color='black')
    ax.vlines(alpha_f, 0, det.max(), color='black', linestyle='--')
    plt.show()


    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
    b = np.transpose(np.array([1, 2, 3], dtype=float))
    x = np.transpose(np.array([1, 2, 2], dtype=float))
    e = 0.01
    imax = 100
    alph_arr = np.arange(0, 1, 0.05)
    det = np.zeros(len(alph_arr))
    k = 0
    alpha = 0
    while alpha < 1:
        x, d, i = const_step_method(A, b, x, e, imax, alpha)
        alpha += 0.05
        det[k] = d
        k += 1
    alpha_f = np.transpose(b - A.dot(x)).dot(b - A.dot(x)) / (np.transpose(b - A.dot(x)).dot(A.dot(b - A.dot(x))))
    fig, ax = plt.subplots()
    plt.title("Graph for matrix 3*3")
    plt.xlabel("Alpha")
    plt.ylabel("Residual")
    plt.yscale('log')
    ax.vlines(alpha_f, 0, det.max(), color='black', linestyle='--')
    ax.plot(alph_arr, det, color='black')
    plt.show()


    A = np.eye(4) + np.diag(np.ones(3), k=1) + np.diag(np.ones(2), k=-2) + np.diag(np.ones(2), k=2) + np.diag(
        np.ones(1), k=3) + np.diag(np.ones(1), k=-3)
    b = np.transpose(np.array([3, 3, 3, 5], dtype=float))
    x = np.transpose(np.array([4, -1, 4, -1], dtype=float))
    e = 0.01
    imax = 50
    alph_arr = np.arange(0, 1, 0.05)
    det = np.zeros(len(alph_arr))
    k = 0
    for alpha in alph_arr:
        x, d, i = const_step_method(A, b, x, e, imax, alpha)
        det[k] = d
        k += 1
    alpha_f = np.transpose(b - A.dot(x)).dot(b - A.dot(x)) / (np.transpose(b - A.dot(x)).dot(A.dot(b - A.dot(x))))
    fig, ax = plt.subplots()
    plt.title("Graph for matrix 4*4")
    plt.xlabel("Alpha")
    plt.ylabel("Residual")
    plt.yscale('log')
    ax.vlines(alpha_f, 0, det.max(), color='black', linestyle='--')
    ax.plot(alph_arr, det, color='black')
    plt.show()


def visualisation(A, b, x, imax, e):
    x1 = np.arange(-5, 5.01, 0.1)
    x2 = np.arange(-5, 5.01, 0.1)

    fig, ax = plt.subplots()
    z = np.zeros((len(x1), len(x1)))
    c = 0
    i, j = 0, 0
    for x10 in x1:
        for x20 in x2:
            x0 = np.transpose(np.array([x10, x20]))
            z[i, j] = 0.5 * np.transpose(x0).dot(A).dot(x0) - np.transpose(b).dot(x0) - c
            i += 1
        j += 1
        i = 0
    lev = np.arange(1, 50, 5)
    ax.contour(x1, x2, z, levels=lev, colors='gray')
    ax.grid()
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')

    print("Method of conjugate gradients")
    x_grad, d_grad, i_grad, x_array_grad = conjugate_gradients_method(A, b, x, e, imax)
    print("Approximate x values: ", x_grad, ", norm value: ", d_grad, ", number of iterations: ", i_grad)
    print("All approximations:", x_array_grad)
    x_grad_graph = np.array([x[0]], dtype=float)
    y_grad_graph = np.array([x[1]], dtype=float)
    for c in x_array_grad:
        x_grad_graph = np.append(x_grad_graph, c[0])
        y_grad_graph = np.append(y_grad_graph, c[1])
    plt.plot(x_grad_graph, y_grad_graph, 'o-', color='black', alpha=0.8)

    print("Method of steepest decent")
    x_decent, d_decent, i_decent, x_array_decent = var_step_method_visualisation(A, b, x, e, imax)
    print("Approximate x values: ", x_decent, ", norm value: ", d_decent, ", number of iterations: ", i_decent)
    print("All approximations:", x_array_decent)
    x_decent_graph = np.array([x[0]], dtype=float)
    y_decent_graph = np.array([x[1]], dtype=float)
    for c in x_array_decent:
        x_decent_graph = np.append(x_decent_graph, c[0])
        y_decent_graph = np.append(y_decent_graph, c[1])
    plt.plot(x_decent_graph, y_decent_graph, 'o--', color='black', alpha=0.8)
    plt.show()

def visualisation_Newton_Raphson(A, b, x, imax, e, jmax, epsilon):
    x1 = np.arange(-5, 5.01, 0.1)
    x2 = np.arange(-5, 5.01, 0.1)

    fig, ax = plt.subplots()
    z = np.zeros((len(x1), len(x1)))
    c = 0
    i, j = 0, 0
    for x10 in x1:
        for x20 in x2:
            x0 = np.transpose(np.array([x10, x20]))
            z[i, j] = 0.5 * np.transpose(x0).dot(A).dot(x0) - np.transpose(b).dot(x0) - c
            i += 1
        j += 1
        i = 0
    lev = np.arange(1, 50, 5)
    ax.contour(x1, x2, z, levels=lev, colors='gray')
    ax.grid()
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')

    f1 = open('output', 'w')
    f1.write("Method of Newton-Raphson\n")
    x_grad, d_grad, i_grad, x_array_grad = conjugate_gradients_Newton_Raphson(A, b, x, imax, e, jmax, epsilon)
    f1.write("Approximate x values: ")
    f1.write(str(x_grad))
    f1.write("\nNorm value: ")
    f1.write(str(d_grad))
    f1.write("\nNumber of iterations: ")
    f1.write(str(i_grad))
    f1.write("\nAll approximations:")
    f1.write(str(x_array_grad))
    x_grad_graph = np.array([x[0]], dtype=float)
    y_grad_graph = np.array([x[1]], dtype=float)
    for coord in x_array_grad:
        x_grad_graph = np.append(x_grad_graph, coord[0])
        y_grad_graph = np.append(y_grad_graph, coord[1])
    plt.plot(x_grad_graph, y_grad_graph, 'o-', color='black', alpha=0.8)

    f1.write("\nMethod of steepest decent")
    x_decent, d_decent, i_decent, x_array_decent = var_step_method_visualisation(A, b, x, e, imax)
    f1.write("\nApproximate x values: ")
    f1.write(str(x_decent))
    f1.write("\nNorm value: ")
    f1.write(str(d_decent))
    f1.write("\nNumber of iterations: ")
    f1.write(str(i_decent))
    f1.write("\nAll approximations:")
    f1.write(str(x_array_decent))
    x_decent_graph = np.array([x[0]], dtype=float)
    y_decent_graph = np.array([x[1]], dtype=float)
    for coord in x_array_decent:
        x_decent_graph = np.append(x_decent_graph, coord[0])
        y_decent_graph = np.append(y_decent_graph, coord[1])
    plt.plot(x_decent_graph, y_decent_graph, 'o--', color='black', alpha=0.8)
    plt.show()
    f1.close()

def main():
    steepest_decent()
    graphs_steepest_decent()
    A = np.array([[2, 3], [3, 8]], dtype=float)
    b = np.transpose(np.array([3, 10], dtype=float))
    x = np.transpose(np.array([-4, 2], dtype=float))  # [-0.86, 1.57]
    imax = 100
    e = 0.01
    visualisation(A, b, x, imax, e)

    f = open('input', 'r')
    A = np.array(f.readline().rstrip().split(), dtype=float)
    n = len(A)
    A = A.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
    b = np.transpose(np.array(f.readline().rstrip().split(), dtype=float))
    x = np.transpose(np.array(f.readline().rstrip().split(), dtype=float))  # [-0.86, 1.57]
    imax = int(f.readline().rstrip())
    jmax = int(f.readline().rstrip())
    e = float(f.readline().rstrip())
    epsilon = float(f.readline().rstrip())
    visualisation_Newton_Raphson(A, b, x, imax, e, jmax, epsilon)
    f.close()

if __name__ == "__main__":
    main()
