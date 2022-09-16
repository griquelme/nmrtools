import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import measure
from typing import Tuple, Union
from matplotlib.colors import to_rgba
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


def _create_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def _draw_colored_patch(
    segments: np.ndarray, c: Union[np.ndarray, str], **kwargs
) -> LineCollection:
    """
    Creates a colored patch object

    Parameters
    ----------
    segments : array of size n by 2
        Array where each row is a xy coordinate.
    c : array of size n
        Color coordinate for each point.

    Other Parameters
    ----------------
    **kwargs : dict
        Parameter passed to LineCollection constructor.

    Returns
    -------
    LineCollection

    """
    if isinstance(c, str):
        c = to_rgba(c)
        lc = LineCollection(segments, colors=c, **kwargs)
    else:
        lc = LineCollection(segments, **kwargs)
        lc.set_array(c)
    return lc


def add_colorbar(
    fig: plt.Figure,
    location: Tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    cmap: str,
    **kwargs
) -> Tuple[ColorbarBase, plt.Axes]:
    """
    Draws a color bar in a figure.

    Parameters
    ----------
    fig : Figure
    location : (float, float, float, float)
        Left, bottom, width and height of the colorbar, relative to the Figure
        width and height.
    vmin : Minimum of the colorbar scale
    vmax : Maximum of the colorbar scale
    cmap : str
        Named colormap

    Other Parameters
    ----------------
    kwargs : dict
        paramters to pass to ColorbarBase

    Returns
    -------

    """
    cbar_ax = fig.add_axes(location)
    cmap = get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, **kwargs)
    return cbar, cbar_ax


def plot_colored_curve(
    x: np.ndarray, y: np.ndarray, c: Union[np.ndarray, str], ax: plt.Axes, **kwargs
) -> LineCollection:
    """
    Draws a curve with a custom color for each segment.

    Parameters
    ----------
    x : 1D array
    y : 1D array
    c : 1D array or str
        color scale for the plot. A ``str`` with a named color can also be used.
    ax : Axes
        Draws the line into this axis.

    Other Parameters
    ----------------
    **kwargs : dict
       Parameter passed to LineCollection constructor.

    Returns
    -------
    LineCollection

    """
    points = _create_points(x, y)
    lc = _draw_colored_patch(points, c, **kwargs)
    ax.add_artist(lc)
    return lc


def plot_colored_contours(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    C: np.ndarray,
    levels: np.ndarray,
    ax: plt.Axes,
    **kwargs
):
    """
    Draw colored contour lines of a 2D array.

    Parameters
    ----------
    x : 1D array
        x coordinates of `Z`. ``len(x) == N``.
    y : 1D array
        y coordinates of `Z`. ``len(y) == M``.
    Z : (M, N) array-like
        Height values used to draw the contour lines.
    C : (M. N) array-like
        Color scale used for the contour lines.s
    levels : array-like
        Levels at which the contour lines are drawn.
    ax : Axis
        Draws the contours into this axis

    Other Parameters
    ----------------
    kwargs : dict
       Parameter passed to LineCollection constructor.

    """

    for lv in levels:
        contours = measure.find_contours(Z, lv)
        for c in contours:
            # contours are in points units and need to be converted to x and y
            z_x_ind, z_y_ind = c.T
            z_x_ind = z_x_ind.round().astype(int)
            z_y_ind = z_y_ind.round().astype(int)

            # xy coordinates of the contours and color
            x_c = x[z_x_ind]
            y_c = y[z_y_ind]
            if isinstance(C, str):
                c_c = C
            else:
                c_c = C[z_x_ind, z_y_ind]
            points = _create_points(x_c, y_c)
            lc = _draw_colored_patch(points, c_c, **kwargs)
            ax.add_artist(lc)
