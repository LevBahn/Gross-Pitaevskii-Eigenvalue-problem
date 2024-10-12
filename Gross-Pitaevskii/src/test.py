from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    N = 1
    E = (1*(np.pi)**2)/2

    def pde(x, psi):
        psi_x = tf.gradients(psi, x)[0]
        psi_xx = tf.gradients(psi_x, x)[0]
        pde = (1/2)*psi_xx + E*psi
        return pde

    def func(x):
        func = np.sqrt(2)*np.sin(N*np.pi*x)
        return func

    geom = dde.geometry.Interval(0, 1)

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)
    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    geom = dde.geometry.Interval(0, 1)

    # boundary conditions
    #bc1 = dde.DirichletBC(geom, func, lambda _, boundary_l : boundary_l)
    #bc2 = dde.DirichletBC(geom, func, lambda _, boundary_r : boundary_r )
    #bc = dde.PointSetBC(np.array([[0.],[1.]]), np.array([[0.],[0.]]), component=0)

    data = dde.data.TimePDE(
        geom,
        pde,
        [],
        num_domain=100,
        num_boundary=2,
        solution=func,
        num_test=2500,
    )
    net = dde.maps.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
    #net.apply_output_transform(lambda x, y: (1-tf.exp(-x))*(1-tf.exp(x-1))*y) # parametric trick from Jin & Protopapas (2020)
    net.apply_output_transform(lambda x, y: x*(x - 1)*y) #hard constraint

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    main()