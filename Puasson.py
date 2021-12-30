import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt


def var_input():
    x1 = float(input("Введите значение x1: "))
    xn = float(input("Введите значение xn: "))
    y1 = float(input("Введите значение y1: "))
    yn = float(input("Введите значение yn: "))
    n = int(input("Введите значение n: "))
    m = int(input("Введите значение m: "))
    return x1, xn, y1, yn, n, m


def fun_input():
    sigma = str(input("Введите функцию правой части уравнения Пуассона: "))
    g1 = str(input("Введите функцию g1 на левой границе: "))
    g2 = str(input("Введите функцию g2 на правой границе : "))
    g3 = str(input("Введите функцию g3 на верхней границе: "))
    g4 = str(input("Введите функцию g4 на нижней границе: "))
    return sigma, g1, g2, g3, g4


def two_dem_to_one_dem(i, j, n):
    return j * n + i


def border(A, B, n, m, x, y, g1, g2, g3, g4):
    k3 = 0
    for j in range(0, m):
        for i in range(0, n):
            k3 = two_dem_to_one_dem(i, j, n)
            if i == 0:
                A[k3][k3] = 1
                code = compile(g1, "<string>", "eval")
                B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
            if i == n - 1:
                A[k3][k3] = 1
                code = compile(g2, "<string>", "eval")
                B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
            if j == 0:
                A[k3][k3] = 1
                code = compile(g3, "<string>", "eval")
                B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
            if j == m - 1:
                A[k3][k3] = 1
                code = compile(g4, "<string>", "eval")
                B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]})
    return A, B


def inside(A, B, n, m, x, y, dx, dy, sigma):
    k1 = k2 = k3 = k4 = k5 = 0
    for j in range(1, m - 1):
        for i in range(1, n - 1):
            k1 = two_dem_to_one_dem(i - 1, j, n)
            k2 = two_dem_to_one_dem(i + 1, j, n)
            k3 = two_dem_to_one_dem(i, j, n)
            k4 = two_dem_to_one_dem(i, j - 1, n)
            k5 = two_dem_to_one_dem(i, j + 1, n)
            A[k3][k1] = 1 / (dx ** 2)
            A[k3][k2] = 1 / (dx ** 2)
            A[k3][k3] = -2 / (dx ** 2) - 2 / (dy ** 2)
            A[k3][k4] = 1 / (dy ** 2)
            A[k3][k5] = 1 / (dy ** 2)
            code = compile(sigma, "<string>", "eval")
            B[k3] = eval(code, {"np": np, "x": x[i], "y": y[j]}) / -8.85418781762039 * (10 ** (-12))
            #B[k3] = -(2.71828182846 ** (-x[i] ** 2 - y[j] ** 2)) / 8.85418781762039 * 10 ** (-12)
    return A, B


x1, xn, y1, ym, n, m = var_input()
sigma, g1, g2, g3, g4 = fun_input()

A = [[0 for j in range(0, n * m)] for i in range(0, n * m)]
B = [0 for i in range(0, m * n)]

dx = (xn - x1) / (n - 1)
dy = (ym - y1) / (m - 1)

x = [i * dx + x1 for i in range(0, n)]
y = [i * dy + y1 for i in range(0, m)]

A, B = inside(A, B, n, m, x, y, dx, dy, sigma)
A, B = border(A, B, n, m, x, y, g1, g2, g3, g4)
U = solve(A, B)

Uij = []
k = 0
for i in range(n):
    Uij.append([])
    for j in range(m):
        Uij[i].append(U[k])
        k += 1


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")

for i in range(len(y)):
    X = np.array(x)
    Y = np.array([y[i] for j in range(len(y))])
    Z = np.array(U[0 + i * len(y):len(y) + len(y) * i])
    ax.plot3D(X, Y, Z, 'b')

for i in range(len(x)):
    Y = np.array(y)
    X = np.array([x[i] for j in range(len(x))])
    Z = np.array(U[0 + i * len(y):len(y) + len(y) * i])
    ax.plot3D(X, Y, Z, 'b')

print(x)
print(y)
print(U)
plt.show()
