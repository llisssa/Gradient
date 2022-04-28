def const(A, b, x, e, imax):
    """
    Функция с постоянным шагом альфа (а)

    Parameters:
    a - альфа, переменная вещественного типа, задается с клавиатуры, это сам шаг, который приближает полученные х к истинному значению числа х
    r - столбец, разность между столбцом b и произведением матрицы А на столбец х, величина ошибки (расхождения) приближенного равенства (невязка)
    A - матрица коэффициентов
    b - столбец свободных членов
    x - столбец решений
    e - погрешность
    imax - максимальное число шагов
    i - int (целочисленный тип данных), переменная, которая хранит значения следующего объекта
    !!!Дописать что такое d и d0!!!

    Returns:
    Выводится полученный столбец решений x, величина ошибки (расхождения) приближенного равенства и количество шагов

    """
    print("Enter alpha")
    a = float(input())
    r = b - A.dot(x)
    d = np.transpose(r).dot(r)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        x = x + (a * r)
        r = b - A.dot(x)
        d = np.transpose(r).dot(r)
        i += 1
    print(x, r, i, sep="\n")


def var(A, b, x, e, imax):
    """
    Функция с переменным шагом альфа (а), он изменяется в зависимости от величины расхождения

    """
    r = b - A.dot(x)
    d = np.transpose(r).dot(r)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        q = A.dot(r)
        a = d / (np.transpose(r).dot(q))
        x = x + a * r
        r = b - A.dot(x)
        d = np.transpose(r).dot(r)
        i += 1
    print(x, r, i, sep="\n")


import numpy as np
print("Enter 2, 3 or 4 for matrix 2*2, 3*3 or 4*4 respectively")
matrix = int(input())

if matrix == 2:
    A = np.array([[2, -4], [2, 1]], dtype=float)
    b = np.transpose(np.array([-6, 4], dtype=float))
    x = np.transpose(np.array([0.5, 1], dtype=float))  # [1, 2]
    e = 0.1
    imax = 100
elif matrix == 3:
    A = np.array([[1, 1, 1], [2, 3, 0], [1, 2, 1]], dtype=float)
    b = np.transpose(np.array([2, 5, -1], dtype=float))
    x = np.transpose(np.array([6, -2, -1], dtype=float))  # [7, -3, -2]
    e = 0.01
    imax = 100
elif matrix == 4:
    A = np.array([[1, 1, 2, 3], [1, 2, 3, -1], [3, -1, -1, -2], [2, 3, -1, -1]], dtype=float)
    b = np.transpose(np.array([1, -4, -4, -6], dtype=float))
    x = np.transpose(np.array([-2, -2, -1, 0], dtype=float))  # [-1, -1, 0, 1]
    e = 0.01
    imax = 100
else:
    print("Error")

if matrix == 2 or matrix == 3 or matrix == 4:
    print("Enter 1 for a constant step or 2 for a variable step")
    step = int(input())
    if step == 1:
        const(A, b, x, e, imax)
    elif step == 2:
        var(A, b, x, e, imax)
    else:
        print("Error")
