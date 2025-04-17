import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
from shap.plots.colors import blue_rgb, red_rgb
from shap.utils import convert_name
from matplotlib.colors import LinearSegmentedColormap, Colormap

mpl.colormaps.register(
    LinearSegmentedColormap.from_list("perovskiteml", [blue_rgb, red_rgb], N=256),
    name="shap"
)
mpl.colormaps.register(
    LinearSegmentedColormap.from_list("perovskiteml", ["royalblue", "deeppink"], N=256),
    name="perovskiteml"
)

def plot_shap_dependence(
    inds: tuple[str],
    shap_values: shap.Explanation,
    cind: str | None = None,
    cmap: str | Colormap | None = "perovskiteml",
    figsize: tuple = (5,4),
    s=10,
    **kwargs
):
    ind0, ind1 = inds
    fig, ax = plt.subplots(tight_layout=True, figsize=figsize)
    
    indx = convert_name(ind1, shap_values[:, ind0, :], shap_values.feature_names)
    values = shap_values[:, ind0, :][:, indx]
    cvalues = shap_values[:, cind, :][:, indx] if cind else None

    scatter = ax.scatter(
        values.data,
        values.values,
        c=cvalues.data if cind else None,
        cmap=cmap if cind else None,
        s=s,
        **kwargs
    )
    
    if cind:
        cbar = plt.colorbar(scatter, aspect=50, format="%2.1f")
        cbar.set_label(cind)
        cbar.outline.set_visible(False)

    ax.set_xlabel(ind0)
    if ind0 == ind1:
        ax.set_ylabel(f"SHAP value for\n{ind0}")
    else:
        ax.set_ylabel(f"SHAP value for\n{ind0} and {ind1}")
        
    return fig, ax