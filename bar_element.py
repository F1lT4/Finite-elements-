import numpy as np
import sympy as sym
from scipy import linalg
# making a bar element with two nodes
""" the polynomial we will use to simulate the displacements is:
  u = a1 + a2 * x
    ^ y 
    |
    |          u2 
  u1.==========.---> x
   /-----L-----/
   [[u1], [u2]] = [[1, 0], [1, L]] * [a1, a2]
   {d} = [A] {a}
"""


def bar_el(length, elast, area):
    x = sym.symbols('x')
    A_mat = np.array([[1, 0], [1, length]])
    # {a} = [A]^-1 * {d}
    A_matrev = linalg.inv(A_mat)
    # u = [1 x] [A]^-1 {d}
    # [N] = [1 x] [A]^-1 ( shape matrix)
    ex_mat = np.array([1, x])
    n_shape = np.matmul(ex_mat, A_matrev)
    ns_1 = n_shape[0]
    ns_2 = n_shape[1]
    # ε = du / dx
    ns_1dif = sym.diff(ns_1, x)
    ns_2dif = sym.diff(ns_2, x)
    # ε = [Β] * {d} ([B] -> deformation matrix)
    b_mat = np.array([[ns_1dif, ns_2dif]])
    b_trans = b_mat.transpose()
    # STIFFNESS MATRIX K = | BT E B dV
    k_temp = np.matmul(b_trans, b_mat)
    for i in range(len(k_temp)):
        for j in range(len(k_temp)):
            k_temp[i][j] = sym.integrate(k_temp[i][j], (x, 0, length))
    k_st = area * elast * k_temp
    return k_st
