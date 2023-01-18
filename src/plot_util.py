import math

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.container
import matplotlib.patches
import matplotlib.pyplot

# Plotting code adapted from https://github.com/michaelaerni/interpolation_robustness

DEFAULT_PPI = 300.0  # points per PIXEL!
FONT_SIZE_PT = 10.0
FONT_SIZE_TICKS_PT = 8.0
FONT_SIZE_CAPTION_PT = 9.0
FONT_SIZE_NORMAL_PT = 10.0
FONT_SIZE_LARGE_PT = 11.0
TEX_PT_PER_IN = 72.27
LINE_WIDTH_PT = 0.8
FIGURE_WIDTH_PT = 397
GUTTER_WIDTH_PT = 10
COLUMN_WIDTH_PT = 23.9146
FIGURE_WIDTH_FOR_COLUMNS_IN = {
    col: (COLUMN_WIDTH_PT * col + GUTTER_WIDTH_PT * (col - 1)) / TEX_PT_PER_IN
    for col in range(1, 12 + 1)
}
LEGEND_HEIGHT_FOR_ROWS_IN = {
    1: 14.0 / TEX_PT_PER_IN,
    2: 28.0 / TEX_PT_PER_IN,
    4: 42.0 / TEX_PT_PER_IN,
}

# Colors based on https://davidmathlogic.com/colorblind
DEFAULT_COLORMAP = matplotlib.colors.ListedColormap(
    colors=("#1E88E5", "#FFC107", "#B51751", "#81CBE6", "#4AB306", "#004D40"), name="cvd_friendly"
)

MARKER_MAP = ("o", "d", "x", "+")

LINESTYLE_MAP = ("solid", "dashed", "dotted", (0, (3, 1, 1, 1, 1, 1)))  # dash dot dot

SHADING_ALPHA = 0.3

GOLDEN_RATIO = 0.5 * (1.0 + math.sqrt(5))
SILVER_RATIO = 1.0 + math.sqrt(2)


def setup_matplotlib():
    matplotlib.pyplot.rcdefaults()

    # Use colormap which works for people with CVD and greyscale printouts
    matplotlib.cm.register_cmap(cmap=DEFAULT_COLORMAP)

    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "image.cmap": DEFAULT_COLORMAP.name,
            "axes.prop_cycle": matplotlib.rcsetup.cycler("color", DEFAULT_COLORMAP.colors),
            "font.family": "sans-serif",
            "font.sans-serif": ["Open Sans"],
            "figure.dpi": DEFAULT_PPI,
            "axes.titlesize": FONT_SIZE_TICKS_PT,
            "axes.labelsize": FONT_SIZE_TICKS_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "patch.linewidth": LINE_WIDTH_PT,
            "xtick.labelsize": FONT_SIZE_TICKS_PT,
            "ytick.labelsize": FONT_SIZE_TICKS_PT,
            "lines.markersize": 3,
            "scatter.edgecolors": "black",
            "errorbar.capsize": 2,
            "legend.frameon": False,
            "legend.fontsize": FONT_SIZE_TICKS_PT,
            "legend.handlelength": 1.0,
            "legend.borderpad": 0.1,
            "legend.borderaxespad": 0.1,
            "legend.labelspacing": 0.2,
            "legend.loc": "center",
            "savefig.dpi": "figure",
            "savefig.pad_inches": 0.0,
            "savefig.transparent": True,
            "figure.constrained_layout.use": True,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": "#d0d0d0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5 * LINE_WIDTH_PT,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75 * LINE_WIDTH_PT,
        }
    )


def proxy_patch() -> matplotlib.patches.Patch:
    return matplotlib.patches.Patch(color="none")
