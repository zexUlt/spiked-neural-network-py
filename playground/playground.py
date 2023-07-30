import numpy as np
from algorithms import nearest_on_ellipse
from timeit import timeit


if __name__ == '__main__':
    target = np.load('../data/tr_target.npy')
    target2 = np.load('../data/vl_target.npy')

    control = np.load('../data/tr_control.npy')
    control2 = np.load('../data/vl_control.npy')

    np.save('../data/control.npy', np.vstack((control, control2)))

