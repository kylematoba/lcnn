import logging
import os
import sys
import math
import functools
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib
import seaborn
import scipy.optimize


if False:
    # palette = seaborn.color_palette("colorblind", 20)
    palette = seaborn.color_palette("dark", 12)
    seaborn.palplot(palette)

USE_HARDCODED_COLORS = True
COLORS = ("b", "g", "r", "c", "m", "y")
ALPHAS = np.linspace(0, 1, 6)
SEABORN_PALETTE = "colorblind"


Color = Union[str, Tuple[float, float, float]]
Plotlims = Tuple[float, float]
FigAx = Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

logger = logging.getLogger(__name__)


def _vec(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (-1, 1))


def _repeat_to_length_n(x: list, n: int) -> list:
    lenx = len(x)
    times_to_repeat = int(np.ceil(n / lenx))
    repeated = x * times_to_repeat
    return repeated[:n]


def get_default_interactive_backend() -> str:
    assert sys.platform == 'darwin', \
        "Need to generalize this if not running on OSX"
    return 'MacOSX'


def _hacky_binary_contourf(
    ax: matplotlib.axes.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    plot_zz: np.ndarray,
    color: Color,
    alpha: float) -> None:
    ax.contourf(xx, yy, plot_zz,
                levels=[0, 2], colors=[color], alpha=alpha)


def _add_bruteforced_decision_boundary_plot(ax: matplotlib.axes.Axes,
                                            pred_func: Callable,
                                            x: torch.tensor,
                                            y: torch.tensor) -> None:
    device = x.device
    dtype = x.dtype

    dim = x.shape[1]
    assert type(pred_func[-1]) == torch.nn.Linear, "only supporting models like this for now"
    device = pred_func[-1].weight.device

    colors = get_palette_of_length(2)

    s = 2
    alpha = 0.4

    x1lims, x2lims = get_plotlims_from_pointcloud(x)
    x1_min, x1_max = x1lims
    x2_min, x2_max = x2lims

    nsteps1 = 200
    nsteps2 = 200

    # nsteps1 = 160
    # nsteps2 = 160

    step1 = (x1_max - x1_min) / nsteps1
    step2 = (x2_max - x2_min) / nsteps2

    grid1 = torch.arange(x1_min, x1_max, step1)
    grid2 = torch.arange(x2_min, x2_max, step2)
    xx, yy = torch.meshgrid(grid1, grid2, indexing='ij')

    pred_func_arg = torch.vstack((xx.ravel(), yy.ravel())).T.to(device)

    z = pred_func(pred_func_arg)

    pos_zexp = (+1 * z).exp()
    neg_zexp = (-1 * z).exp()
    plot_zz0 = torch.where(pos_zexp < 1.0, pos_zexp, torch.ones_like(z) * math.nan).cpu().detach().reshape(xx.shape)
    plot_zz1 = torch.where(neg_zexp < 1.0, neg_zexp, torch.ones_like(z) * math.nan).cpu().detach().reshape(xx.shape)

    _hacky_binary_contourf(ax, xx, yy, plot_zz0, colors[0], alpha)
    _hacky_binary_contourf(ax, xx, yy, plot_zz1, colors[1], alpha)

    rows0 = (0 == y).flatten()
    rows1 = (1 == y).flatten()

    ax.scatter(x[rows0, 0].cpu().detach(),
               x[rows0, 1].cpu().detach(), color=colors[0], s=s)

    ax.scatter(x[rows1, 0].cpu().detach(),
               x[rows1, 1].cpu().detach(), color=colors[1], s=s)

    # old_calc = True
    old_calc = False
    if old_calc:
        eps = .001
        boundary_rows = z[:, 0].abs() < eps
        boundary_data = pred_func_arg[boundary_rows, :]
    else:
        plot_every = 5
        plot_grid = grid1[:nsteps1:plot_every]
        matched_y = torch.full(plot_grid.shape, math.nan)
        for idx, x1 in enumerate(plot_grid):
            # idx = 0; x1 = grid1[idx]
            f1 = lambda _: pred_func(torch.tensor(np.array([x1.item(), _.item()]), device=device, dtype=dtype)).cpu().detach() ** 2.0
            optim_res = scipy.optimize.root(f1, x1.item())
            matched_y[idx] = optim_res.x.item()
        roots = torch.vstack((plot_grid, matched_y)).T.to(pred_func_arg.device)
        boundary_data = roots

    num_boundary_points = boundary_data.shape[0]
    #ax.scatter(boundary_data[:, 0].cpu().detach(),
    #           boundary_data[:, 1].cpu().detach(), color="k", s=s)

    grads = torch.empty((num_boundary_points, dim))
    for row in range(num_boundary_points):
        xxx = boundary_data[row, :]
        ggg = torch.autograd.functional.jacobian(pred_func, xxx)
        grads[row, :] = ggg
        # print((xxx * ggg).sum())

    qx = boundary_data[:, 0].cpu().detach()
    qy = boundary_data[:, 1].cpu().detach()
    qu = qx + grads[:, 0].cpu().detach()
    qv = qy + grads[:, 1].cpu().detach()
    #ax.quiver(qx, qy, qu, qv)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def get_palette_of_length(num_colors: int) -> list:
    if USE_HARDCODED_COLORS:
        colors = _repeat_to_length_n(COLORS, num_colors)
    else:
        colors = seaborn.color_palette(SEABORN_PALETTE, num_colors)
    return colors


def bruteforced_decision_boundary_plot(x: np.ndarray,
                                       y: np.ndarray,
                                       model: Callable,
                                       plot_scale: float) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, axs = wrapped_subplot(1, 1, plot_scale)
    ax = axs[0, 0]
    _add_bruteforced_decision_boundary_plot(ax, model, x, y)
    # _style_plot(ax)
    return fig, ax


def _matrix_rank_0(x: np.ndarray) -> int:
    """ Matrix rank calculation that naturally handles the empty case """
    if 0 == x.size:
        mr = 0
    else:
        mr = np.linalg.matrix_rank(x)
    return mr


def _convex_hull_plot_kernel(
    ax: matplotlib.axes.Axes,
    points: np.ndarray,
    color: str = "grey",
    alpha: float = 0.2,
) -> None:
    """
    Plots a polytope (no rays), assumes (without checking) that the
    points are distinct [only used in the treatment of 1 and 2 point
    sets, though]
    """
    num_points, point_dim = points.shape
    assert 2 == point_dim

    points_rank = _matrix_rank_0(points)
    if num_points >= 3 and (points_rank >= 2):
        hull = scipy.spatial.ConvexHull(points)
        # point_alpha = 0.5
        point_alpha = 0.0
        ax.plot(points[:, 0], points[:, 1], ".", color=color, alpha=point_alpha)
        centre = np.mean(points, 0)
        pts = []
        for pt in points[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())

        pts.sort(key=lambda p: np.arctan2(p[1] - centre[1], p[0] - centre[0]))
        pts.insert(len(pts), pts[0])
        pts_array = (np.array(pts) - centre) + centre

        poly = matplotlib.patches.Polygon(pts_array,
                                          facecolor=color,
                                          edgecolor=None,
                                          alpha=alpha)
        ax.add_patch(poly)
    elif points.shape[0] > 1:
        ax.plot(points[:, 0], points[:, 1], lw=4, color=color, alpha=alpha)
    else:
        ax.plot(points[:, 0], points[:, 1], ".", lw=5, color=color, alpha=alpha)


def wrapped_subplot(subplot_rows: int,
                    subplot_cols: int,
                    plot_scale: float = 3.0) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    fig, axs = plt.subplots(
        subplot_rows,
        subplot_cols,
        figsize=(subplot_cols * plot_scale, subplot_rows * plot_scale),
    )
    axs = np.reshape(axs, (subplot_rows, subplot_cols))
    return fig, axs


def _style_plot(ax) -> None:
    major_axis_grid_color = "black"
    # major_axis_grid_color = "grey"

    axline_lw = 0.50
    major_grid_lw = .50
    minor_grid_lw = .25

    do_minor_ticks = False
    if do_minor_ticks:
        ax.minorticks_on()
        ax.grid(which="minor", color="grey", linewidth=minor_grid_lw)
    ax.grid(which="major", color=major_axis_grid_color, linewidth=major_grid_lw)
    ax.axhline(y=0, color=major_axis_grid_color, linewidth=axline_lw)
    ax.axvline(x=0, color=major_axis_grid_color, linewidth=axline_lw)


def _nudge_out_limits(lim: Plotlims, factor: float) -> Plotlims:
    # Add a bit in each direction to better visualise around
    # the edge of the plot
    assert 0 <= factor <= 0.5, "factor should be a small positive number"
    limdiff = lim[1] - lim[0]

    nudge_out_by = limdiff * factor
    assert math.isfinite(nudge_out_by)
    assert not math.isnan(nudge_out_by)
    return lim[0] - nudge_out_by, lim[1] + nudge_out_by


def get_plotlims_from_v_form(v: np.ndarray,
                             factor: float) -> Tuple[Plotlims, Plotlims]:
    if 0 == v.size:
        mins = (+1 * np.inf, +1 * np.inf)
        maxs = (-1 * np.inf, -1 * np.inf)
    else:
        # v = np.atleast_2d(v)
        mins = np.min(v[:, 1:], axis=0)
        maxs = np.max(v[:, 1:], axis=0)

    xlim_direct = (mins[0], maxs[0])
    ylim_direct = (mins[1], maxs[1])

    xlim = _nudge_out_limits(xlim_direct, factor)
    ylim = _nudge_out_limits(ylim_direct, factor)
    return xlim, ylim


def get_plotlims_from_pointcloud(xy: np.ndarray) -> Tuple[Plotlims, Plotlims]:
    nr, nc = xy.shape
    assert nr > 0
    assert nc == 2

    mins = xy.min(axis=0).values
    maxs = xy.max(axis=0).values

    x1lim_direct = (mins[0].item(), maxs[0].item())
    x2lim_direct = (mins[1].item(), maxs[1].item())

    factor = 0.15
    x1lim = _nudge_out_limits(x1lim_direct, factor)
    x2lim = _nudge_out_limits(x2lim_direct, factor)
    return x1lim, x2lim


def initialise_pgf_plots(texsystem: str, font_family: str) -> None:
    plt.switch_backend("pgf")
    # https://matplotlib.org/users/customizing.html
    pgf_with_rc_fonts = {
        "pgf.texsystem": texsystem,
        "font.family": font_family,
        "font.serif": [],
        "text.usetex": True,
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)


def finalize_plot() -> None:
    default_interactive_backend = get_default_interactive_backend()
    plt.switch_backend(default_interactive_backend)


def smart_save_fig(fig: matplotlib.figure.Figure,
                   ident: str,
                   fig_format: str,
                   filepath: str) -> str:
    filename = "{}.{}".format(ident, fig_format)
    os.makedirs(filepath, exist_ok=True)
    fig_path = os.path.join(filepath, filename)
    fig.savefig(fig_path, bbox_inches="tight")
    print(fig_path)
    return fig_path


if __name__ == "__main__":
    xyz = np.random.rand(5, 3)
