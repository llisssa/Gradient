def const(A, b, x, e, imax):
    """
    A - матрица коэффициентов
    b - столбец свободных членов
    x - столбец решений
    """
    a = float(input())
    r = b - A.dot(x)
    d = np.transpose(r).dot(r)
    d0 = d
    i = 0
    while (i < imax) and (d > e**2 * d0):
        x = x + a*r
        r = b - A.dot(x)
        d = np.transpose(r).dot(r)
        i += 1
    print(x, r, i)


import numpy as np
matrix = int(input())
if matrix == 2:
    A = np.array([[2, -4], [2, 1]], dtype=float)
    b = np.transpose(np.array([-6, 4], dtype=float))
    x = np.transpose(np.array([0.5, 1], dtype=float))
    e = 0.1
    imax = 100
    const(A, b, x, e, imax)
elif matrix == 3:
    A = np.array([[1, 1, 1], [2, 3, 0], [1, 2, 1]], dtype=float)
    b = np.transpose(np.array([2, 5, -1], dtype=float))
    x = np.transpose(np.array([6, -2, -2], dtype=float))
    e = 0.01
    imax = 100



