"""Shared matplotlib rcParams so every figure in the report has a
consistent look. Import this module to apply."""
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "lines.linewidth":   1.5,
})
