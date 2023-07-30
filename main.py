# -*- coding: utf-8 -*-

from enum import IntEnum, auto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from activation_functions import Sigmoid
from data_generator import DataGenerator
from data_producer import IzhikevichProducer
from projectors import EllipsoidProjector
from gamma_func import MultiplicativeGamma
from models.v2 import ProjectedSDNN

plt.style.use(['science', 'ieee', 'high-vis'])
mpl.rc('text', usetex=False)
mpl.rc('axes', linewidth=2)
mpl.rc('font', weight='bold')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams["axes.labelweight"] = "bold"


def uu(t):
    return np.array([[np.exp(np.sin(t)), np.exp(np.sin(t)) * (1 - np.cos(t) + np.sin(t))]]).T


def xx(t):
    return np.array([[np.exp(np.sin(t)) * (1 - np.cos(t) + np.sin(t)),
                      np.exp(np.sin(t))]]).T


def low_pass_filter(x, border=(0, 0.99)):
    length, ncols = x.shape
    frequencies = np.fft.rfftfreq(length, d=1e-2)
    min_f, max_f = border

    for i in range(ncols):
        fourier = np.fft.rfft(x[:, i])
        ft_threshed = fourier.copy()
        ft_threshed[(min_f >= frequencies)] = 0
        ft_threshed[(max_f <= frequencies)] = 0
        ifourier = np.fft.irfft(ft_threshed, length)
        x[:, i] = ifourier.copy()


def initialize_base_vars(x_shape: int, u_shape: int, use_izh=False, from_file: dict = None, unfiltered: bool = False):
    step = 1e-4
    time_end = 4.8 * np.pi
    if from_file is None:
        time = np.arange(0, time_end, step)

        producer = IzhikevichProducer((1, x_shape))
        data_gen = DataGenerator(n=len(time), producer=producer)

        if not use_izh:
            x_in = xx(time)
            u_in = uu(time)
        else:
            x_in = data_gen.generate(time).reshape((-1, x_shape, 1))
            u_in = np.ones(shape=(time.shape[0], u_shape, 1)) * 15.
    else:
        x_in = np.load(from_file['x'])
        if not unfiltered:
            low_pass_filter(x_in)
        x_in = x_in.reshape((-1, x_in.shape[1], 1))[:-int(0.2 * x_in.shape[0])]

        u_in = np.load(from_file['u'])
        u_in = np.diff(u_in, axis=0) * 120
        if not unfiltered:
            low_pass_filter(u_in)
        u_in = u_in.reshape((-1, u_in.shape[1], 1))[:-int(0.2 * u_in.shape[0])]

        time = np.arange(0, time_end, time_end / x_in.shape[0])

    return time_end, step, x_in[:-1], u_in, time[:-1]


def draw_smth(time: np.array, x_in: np.array, x_pred: np.array, loss: np.array, w_a: np.array, w_b: np.array, stop_time,
              unfiltered: bool):
    fig, ax = plt.subplots(nrows=1, figsize=(3, 3), dpi=190, sharex=True, sharey=False)
    ax.plot(time, x_in[:, 0, :], label=r"$\mathbf{x_0}$")
    ax.plot(time, x_pred[:, 0, :], label=r"$\mathbf{\hat x_0}$")
    ax.set_xlabel(r'Time (s)', fontsize='x-large')
    ax.set_ylabel(r'Yaw angle (rad)', fontsize='x-large')
    ax.legend(loc='upper left', frameon=True)
    # ax.autoscale(tight=True)
    ax.tick_params(labelsize='x-large')
    ax.axvline(x=stop_time, color='green', linestyle=':') if stop_time < time_end else None

    if stop_time < time_end:
        axins = inset_axes(ax, width=1, height=1, loc='upper right', bbox_to_anchor=(0.85, 1.2),
                           bbox_transform=ax.figure.transFigure, axes_kwargs={'zorder': 10})
        bbox = axins.get_tightbbox()
        x, y, w, h = bbox.bounds
        fig.patches.extend([FancyBboxPatch((0.67, 1.06), 0.001, 0.001, boxstyle='round,rounding_size=0.05',
                                          fill=True, facecolor='#e4c9a9', alpha=0.8, zorder=5, linewidth=2,
                                          edgecolor='#4D4D4D', transform=fig.transFigure, figure=fig)])
        # ax.add_patch(Rectangle((0, 0), w + 10, h + 10, facecolor='red', fill=True))
        extent = (stop_time - 0.5, stop_time + 0.7)
        axins.plot(time, x_in[:, 0, :])
        axins.plot(time, x_pred[:, 0, :])
        axins.set_xlim(extent[0], extent[1])
        axins.set_ylim(-0.07, 0.05)
        axins.axvline(x=stop_time, color='green', linestyle=':')
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="black", zorder=10)
    plt.savefig(f'./plots/vor_estimation_L'
                f'{"_fixed" if stop_time < time_end else ""}'
                f'{"_unfiltered" if unfiltered else ""}.svg')

    # LOSS
    loss_norm = np.linalg.norm(loss, axis=1)
    fig, ax = plt.subplots(nrows=1, figsize=(3, 3), dpi=190, sharex=True, sharey=False)
    ax.plot(time[:-1], loss_norm.flatten())
    ax.set_ylabel(r"$\mathbf{||\Delta||}$", fontsize='xx-large')
    ax.set_xlabel(r'Time (s)', fontsize='x-large')
    ax.autoscale(tight=True)
    if not unfiltered and stop_time > time_end:
        ax.set_ylim(0, 0.0035)
    ax.tick_params(labelsize='x-large')
    ax.axvline(x=stop_time, color='red', linestyle='--') if stop_time < time_end else None
    plt.savefig(f'./plots/vor_loss{"_fixed" if stop_time < time_end else ""}{"_unfiltered" if unfiltered else ""}.svg')

    # WEIGHTS 1
    fig, ax = plt.subplots(nrows=2, figsize=(3, 5), dpi=190, sharex=True, sharey=False)
    ax[0].plot(time[:-1], np.linalg.norm(w_a, axis=(1, 2))[:-1])
    ax[0].set_ylabel(r"$\mathbf{||W_1||_F}$", fontsize='x-large')
    ax[0].set_xlabel(r'Time (s)')
    # ax.autoscale(tight=True)
    ax[0].tick_params(labelsize='x-large')
    ax[0].axvline(x=stop_time, color='red', linestyle='--') if stop_time < time_end else None
    # plt.savefig(f'./plots/vor_weights_1'
    #             f'{"_fixed" if stop_time < time_end else ""}'
    #             f'{"_unfiltered" if unfiltered else ""}.svg')

    # WEIGHTS 2
    # fig, ax = plt.subplots(nrows=1, figsize=(3, 3), dpi=190, sharex=True, sharey=False)
    ax[1].plot(time[:-1], np.linalg.norm(w_b, axis=(1, 2))[:-1])
    ax[1].set_ylabel(r"$\mathbf{||W_2||_F}$", fontsize='x-large')
    ax[1].set_xlabel(r'Time (s)')
    ax[1].tick_params(labelsize='x-large')
    # ax.autoscale(tight=True)
    ax[1].axvline(x=stop_time, color='red', linestyle='--') if stop_time < time_end else None
    fig.align_ylabels(ax)
    plt.savefig(f'./plots/vor_weights'
                f'{"_fixed" if stop_time < time_end else ""}'
                f'{"_unfiltered" if unfiltered else ""}.svg')


def get_variables_shapes(dim1, dim2, state_size, u_size):
    b1 = dim1
    c1 = (state_size, dim1)
    d1 = dim1
    e1 = dim1

    b2 = (dim2, u_size)
    c2 = (state_size, dim2, u_size)
    d2 = (dim2, u_size)
    e2 = (dim2, u_size)

    return b1, c1, d1, e1, b2, c2, d2, e2


def init_dnn(params: dict):
    np.random.seed(2)

    # VOR
    b1 = np.ones(b1_shape) * -0.5
    c1 = np.ones(c1_shape) * -0.001
    d1 = np.ones(d1_shape) * -0.5
    b2 = np.ones(b2_shape) * -0.15
    c2 = np.ones(c2_shape) * -0.001
    d2 = np.ones(d2_shape) * -0.5

    # b1 = np.ones(b1_shape) * -0.5
    # c1 = np.ones(c1_shape) * -0.01
    # d1 = np.ones(d1_shape) * -1
    # b2 = np.ones(b2_shape) * -0.5
    # c2 = np.ones(c2_shape) * -0.01
    # d2 = np.ones(d2_shape) * -1

    nn1 = params['nn1']
    nn2 = params['nn2']
    u_size = params['u_size']

    sigma_1 = Sigmoid(nn1, 1, b1, c1, d1)
    sigma_2 = Sigmoid(nn2, u_size, b2, c2, d2)

    params['sigma_1'] = sigma_1
    params['sigma_2'] = sigma_2

    gamma_2 = MultiplicativeGamma(param=params['c_2'], is_inner=False, weight_matrix=params['p_external'])
    gamma_1 = MultiplicativeGamma(param=params['c_1'], is_inner=True,
                                  weight_matrix=params['p_external'] / params['beta'])

    params['gamma_1'] = gamma_1
    params['gamma_2'] = gamma_2

    eps = 0.0001
    shape = params['p_external'].shape
    w1_proj = EllipsoidProjector(params['p_external'] / params['beta'] + eps * np.eye(*shape))
    w2_proj = EllipsoidProjector(params['p_external'] - eps * np.eye(*shape))

    params['w1_projector'] = w1_proj
    params['w2_projector'] = w2_proj

    s_dnn = ProjectedSDNN(**params)

    return s_dnn


def process(model):
    model.fit(x=x_input, u=u_input, step=1e-4)

    loss_norm = np.sum(np.linalg.norm(np.asarray(model.history_loss), axis=1))
    loss_delta = np.sum(np.asarray(list(map(lambda el: el.T @ el, model.history_loss))))

    print(f"Loss Norm:\t{loss_norm}")
    print(f"Loss Delta:\t{loss_delta}")

    return model


def run_test_1(model, unfiltered: bool = False):
    model = process(model)

    x = np.array(model.history_x)
    loss = np.array(model.history_loss)

    fig, ax = plt.subplots(nrows=3, figsize=(3, 5), dpi=190, sharex=True, sharey=False)
    fig.canvas.draw()
    ax[0].plot(time, u_input[:, 0, 0])
    ax[0].set_ylabel(r'$\mathbf{u_x}$ (rad/s)', fontsize='x-large')
    ax[0].autoscale(tight=True)

    ax[1].plot(time, u_input[:, 1, 0])
    ax[1].autoscale(tight=True)
    ax[1].set_ylabel(r'$\mathbf{u_z}$ (rad/s)', fontsize='x-large')

    ax[2].plot(time, u_input[:, 2, 0])
    ax[2].autoscale(tight=True)
    ax[2].set_ylabel(r'$\mathbf{u_y}$ (rad/s)', fontsize='x-large')
    ax[2].set_xlabel(r'Time (s)', fontsize='x-large')
    fig.align_ylabels(ax)
    plt.savefig(f'./plots/vor_control{"_unfiltered" if unfiltered else ""}.svg')

    draw_smth(time=time, x_in=x_input, x_pred=x, loss=loss, w_a=np.asarray(model.history_Wa),
              w_b=np.asarray(model.history_Wb), stop_time=model.stop_time, unfiltered=unfiltered)

    # noinspection PyTypeChecker

    s_dnn_sigma = np.array(model.history_sigma)
    s_dnn_phi_u = np.array(model.history_phi_U)

    sigma_1_norm = np.linalg.norm(s_dnn_sigma, axis=1)
    sigma_2_norm = np.linalg.norm(s_dnn_phi_u, axis=1)
    if not unfiltered:
        fig, ax = plt.subplots(nrows=2, figsize=(3, 5), dpi=190, sharex=True, sharey=False)
        ax[0].plot(time[:-1], sigma_1_norm, label=r'$|W_1\sigma_1(x)|$')
        # ax[0].plot(time[:-1], s_dnn_sigma[:, 1], label=r'$(W_1\sigma_1(x))_2$')
        ax[0].legend(frameon=True, fontsize='x-large')
        ax[0].tick_params(labelsize='x-large')
        ax[0].axvline(x=model.stop_time, linestyle='-.', color='black') if model.stop_time < time_end else None
        ax[0].autoscale(tight=True)

        ax[1].plot(time[:-1], sigma_2_norm, label=r'$|W_2\sigma_2(x)u|$')
        # ax[1].plot(time[:-1], s_dnn_phi_u[:, 1], label=r'$(W_2\sigma_2(x)u)_2$')
        ax[1].legend(frameon=True, fontsize='x-large')
        ax[1].axvline(x=model.stop_time, linestyle='-.', color='black') if model.stop_time < time_end else None
        ax[1].autoscale(tight=True)
        ax[1].tick_params(labelsize='x-large')
        ax[1].set_xlabel(r'Time (s)', fontsize='x-large')
        plt.savefig(f'./plots/vor_activation{"_fixed" if model.stop_time < time_end else ""}.svg')


class EDataSource(IntEnum):
    SinusoidalFunction = auto()
    Izhikevich = auto()
    VOR = auto()


def get_best_params(data_source=EDataSource.SinusoidalFunction):
    np.random.seed(2)
    match data_source:
        case EDataSource.Izhikevich:
            return {
                'state_size': state_size,
                'u_size': u_size,
                'k1': 7,
                'k2': 7,
                'nn1': nn1,
                'nn2': nn2,
                'w1_init': np.random.rand(state_size, nn1) * 1e-1,
                'w2_init': np.random.rand(state_size, nn2) * 20e1,
                'a_init': np.identity(2) * -50.,
                'p_init': np.array([[1000, 100.], [100., 1000]]) * 1,
                'l': np.array([[1500, 0.], [0., 1500]]) * 3.5,
                'p_external': np.eye(state_size * nn2, dtype=float) * 0.001,
                'beta': 0.4,
                'c_1': 1,
                'c_2': 0.05,
                'stop_time': time_end - 2 * np.pi,
                'sampling_step': step
            }
        case EDataSource.SinusoidalFunction:
            return {
                'state_size': state_size,
                'u_size': u_size,
                'k1': 0.01,  # 0.01,
                'k2': 0.01,  # 0.01,
                'nn1': nn1,
                'nn2': nn2,
                'w1_init': np.random.rand(state_size, nn1) * 1e-1,
                'w2_init': np.random.rand(state_size, nn2) * 7e1,
                'a_init': np.identity(2) * -50.,
                'p_init': np.array([[60, 20.], [20., 80]]) * 10,
                'l': np.array([[800, 0.], [0., 900]]) * 0.1,
                'p_external': np.eye(state_size * nn2, dtype=float) * 0.1,
                'beta': 0.4,
                'c_1': 1,
                'c_2': 0.05,
                'stop_time': time_end - 2 * np.pi,
                'sampling_step': step
            }
        case EDataSource.VOR:
            return {
                'state_size': state_size,
                'u_size': u_size,
                'k1': 0.01,  # 0.01,
                'k2': 10,  # 0.01,
                'nn1': nn1,
                'nn2': nn2,
                'w1_init': np.random.rand(state_size, nn1) * 1e-2,
                'w2_init': np.random.rand(state_size, nn2) * 1.25e1,
                'a_init': np.identity(2) * -50.,
                'p_init': np.array([[6, 2.], [2., 8]]) * 10,
                'l': np.array([[500, 0.], [0., 800]]) * 15,
                'p_external': np.eye(state_size * nn2, dtype=float) * 0.005,
                'beta': 0.1,
                'c_1': 0.05,
                'c_2': 0.05,
                'stop_time': 9,
                'sampling_step': time_end / x_input.shape[0]
            }


time_end = ...
x_input = ...
u_input = ...
step = ...
time = ...


if __name__ == '__main__':
    nn1 = 2
    nn2 = 2
    state_size = 2
    u_size = 2

    data = {
        'x': './data/target.npy',
        'u': './data/control.npy'
    }

    # data = None
    unfiltered = False
    time_end, step, x_input, u_input, time = initialize_base_vars(x_shape=state_size, u_shape=u_size, use_izh=True,
                                                                  from_file=data, unfiltered=unfiltered)
    state_size = x_input.shape[1]
    u_size = u_input.shape[1]

    b1_shape, c1_shape, d1_shape, e1_shape, \
        b2_shape, c2_shape, d2_shape, e2_shape = get_variables_shapes(dim1=nn1, dim2=nn2,
                                                                      state_size=state_size,
                                                                      u_size=u_size)
    dnn_params = get_best_params(data_source=EDataSource.VOR)

    model = init_dnn(dnn_params)

    run_test_1(model, unfiltered)
