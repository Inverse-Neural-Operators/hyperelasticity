# external imports
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
import matplotlib.tri as mtri
import seaborn as sns

sns.set_theme(style="white")

tab20 = plt.cm.tab20.colors
tab20_blue = tab20[0]
tab20_orange = tab20[2]
tab20_green  = tab20[4]
tab20_red    = tab20[6]

I1bar_min = 3.0
I2bar32_min = 3.0**(3/2)

def Wbar_admissible_domain(Wbar1=None, Wbar2=None, Ibar_range = 3, contour_discretize = 100):
    # contour plot
    I1bar_max = I1bar_min + Ibar_range
    I2bar32_max = I2bar32_min + Ibar_range
    x = np.linspace(I1bar_min, I1bar_max, contour_discretize) # invariant I1bar
    y = np.linspace(I2bar32_min, I2bar32_max, contour_discretize) # invariant I2bar32
    X, Y = np.meshgrid(x, y)
    if Wbar1 is not None:
        Z1 = Wbar1(X, Y)
    else:
        Z1 = np.zeros_like(X)
    if Wbar2 is not None: Z2 = Wbar2(X, Y)
    X = X - I1bar_min # invariant I1bar - 3
    Y = Y - I2bar32_min # invariant I2bar32 - 3^(3/2)

    # admissible domain
    t = np.linspace(1/20, 20, 500)
    I1bar_curve = t**2 + 2*t**(-1) - I1bar_min # invariant I1bar - 3
    I2bar32_curve = (t**(-2) + 2*t)**(3/2) - I2bar32_min # invariant I2bar32 - 3^(3/2)
    Ibar_curve = np.column_stack([I1bar_curve, I2bar32_curve])
    boundary = np.vstack([Ibar_curve])
    path = Path(boundary)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(points).reshape(X.shape)
    Z1_masked = np.where(mask, Z1, np.nan)
    if Wbar2 is not None: Z2_masked = np.where(mask, Z2, np.nan)

    if Wbar2 is None: return X, Y, Z1_masked, Ibar_curve
    else: return X, Y, Z1_masked, Z2_masked, Ibar_curve

def plot_Wbar_I1bar_I2bar32(
        Wbar1=None,
        Wbar2=None,
        name_Wbar1="truth",
        name_Wbar2="discovered",
        color_Wbar1=tab20_blue,
        color_Wbar2=tab20_orange,
        Ibar_range = 3,
        scale_I1bar = 1.0,
        scale_I2bar32 = 1.0,
        contour_discretize = 100,
        contour_levels = 11,
        ):
    
    # evaluate isochoric strain energy density
    # if Wbar1 is not None:
    if Wbar2 is None:
        X, Y, Z_masked, Ibar_curve = Wbar_admissible_domain(
            Wbar1,
            Ibar_range = Ibar_range,
            contour_discretize = contour_discretize
        )
    else:
        X, Y, Z1_masked, Z2_masked, Ibar_curve = Wbar_admissible_domain(
            Wbar1,
            Wbar2,
            Ibar_range = Ibar_range,
            contour_discretize = contour_discretize
        )

    # plot
    fig, ax = plt.subplots()
    if Wbar1 is not None:
        if Wbar2 is None:
            levels = np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), contour_levels)
            contours = ax.contour(X, Y, Z_masked, levels=levels, colors=color_Wbar1)
            ax.clabel(
                contours,
                inline=True,
                fontsize=8,
                fmt="%.2f" # format numbers
            )
            legend_elements = [
                Line2D([0], [0], color="gray", lw=2, linestyle="--", label="UT / BT"),
                Line2D([0], [0], color=color_Wbar1, lw=2, label=fr"$\bar W$ ({name_Wbar1})"),
            ]
        else:
            levels = np.linspace(np.nanmin(Z1_masked), np.nanmax(Z1_masked), contour_levels)
            contours1 = ax.contour(X, Y, Z1_masked, levels=levels, colors=color_Wbar1)
            contours2 = ax.contour(X, Y, Z2_masked, levels=levels, colors=color_Wbar2, linestyles='--')
            ax.clabel(
                contours1,
                inline=True,
                fontsize=8,
                fmt="%.2f" # format numbers
            )
            ax.clabel(
                contours2,
                inline=True,
                fontsize=8,
                fmt="%.2f" # format numbers
            )
            legend_elements = [
                Line2D([0], [0], color="gray", lw=2, linestyle="--", label="UT / BT"),
                Line2D([0], [0], color=color_Wbar1, lw=2, label=fr"$\bar W$ ({name_Wbar1})"),
                Line2D([0], [0], color=color_Wbar2, lw=2, linestyle="--", label=fr"$\bar W$ ({name_Wbar2})"),
            ]
    else:
        legend_elements = [
            Line2D([0], [0], color="gray", lw=2, linestyle="--", label="UT / BT"),
        ]
    ax.plot(Ibar_curve[:, 0], Ibar_curve[:, 1], "--", linewidth=2, color="gray")
    ax.legend(handles=legend_elements)
    ax.set_xlim(0.0, scale_I1bar*Ibar_range)
    ax.set_ylim(0.0, scale_I2bar32*Ibar_range)
    plt.xlabel(r"$\bar{I}_1 - 3$")
    plt.ylabel(r"$\bar{I}_2^{3/2} - 3^{3/2}$")
    ax.set_box_aspect(1)
    
    return fig, ax, legend_elements

def plot_Wbar_I1bar_I2bar32_field(
        Wbar1,
        Wbar2=None,
        name_Wbar1="truth",
        name_Wbar2="discovered",
        Ibar_range=3,
        scale_I1bar=1.0,
        scale_I2bar32=1.0,
        contour_discretize=100,
        contour_levels=11,
        cmap="viridis",
        ):

    # evaluate isochoric strain energy density
    if Wbar2 is None:
        X, Y, Z_masked, Ibar_curve = Wbar_admissible_domain(
            Wbar1,
            Ibar_range=Ibar_range,
            contour_discretize=contour_discretize
        )
    else:
        X, Y, Z1_masked, Z2_masked, Ibar_curve = Wbar_admissible_domain(
            Wbar1,
            Wbar2,
            Ibar_range=Ibar_range,
            contour_discretize=contour_discretize
        )

    # plot
    fig, ax = plt.subplots(figsize=(3, 3))

    if Wbar2 is None:

        field = ax.pcolormesh(
            X,
            Y,
            Z_masked,
            cmap=cmap,
            shading="auto",
        )

        cbar = fig.colorbar(field, ax=ax)
        cbar.set_label(r"$\bar W$")

        legend_elements = [
            Line2D(
                [0], [0],
                color="gray",
                lw=2,
                linestyle="--",
                label="UT / BT"
            ),
        ]

    else:

        vmin = np.nanmin([np.nanmin(Z1_masked),
                          np.nanmin(Z2_masked)])
        vmax = np.nanmax([np.nanmax(Z1_masked),
                          np.nanmax(Z2_masked)])

        field = ax.pcolormesh(
            X,
            Y,
            Z1_masked,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        cbar = fig.colorbar(field, ax=ax)
        cbar.set_label(fr"$\bar W$ ({name_Wbar1})")

        levels = np.linspace(vmin, vmax, contour_levels)

        contours2 = ax.contour(
            X,
            Y,
            Z2_masked,
            levels=levels,
            colors="white",
            linewidths=1.5,
            linestyles="--",
        )

        ax.clabel(
            contours2,
            inline=True,
            fontsize=8,
            fmt="%.2f",
        )

        legend_elements = [
            Line2D(
                [0], [0],
                color="gray",
                lw=2,
                linestyle="--",
                label="UT / BT"
            ),
            Line2D(
                [0], [0],
                color="white",
                lw=2,
                linestyle="--",
                label=fr"$\bar W$ ({name_Wbar2})"
            ),
        ]

    # admissible boundary
    ax.plot(
        Ibar_curve[:, 0],
        Ibar_curve[:, 1],
        "--",
        linewidth=2,
        color="gray",
    )

    # ax.legend(handles=legend_elements)

    ax.set_xlim(0.0, scale_I1bar * Ibar_range)
    ax.set_ylim(0.0, scale_I2bar32 * Ibar_range)

    ax.set_xlabel(r"$\bar{I}_1 - 3$")
    ax.set_ylabel(r"$\bar{I}_2^{3/2} - 3^{3/2}$")

    ax.set_box_aspect(1)

    return fig, ax, legend_elements

def plot_Wbar_lam1_lam2(
        Wbar1,
        Wbar2=None,
        theta2=None,
        contour_discretize = 100,
        contour_levels = 11,
        ):
    
    lam1bar_min = 1.0
    lam2bar_max = 1.25
    x = np.linspace(lam1bar_min, lam2bar_max, contour_discretize)
    y = np.linspace(lam1bar_min, lam2bar_max, contour_discretize)
    X, Y = np.meshgrid(x, y)
    lam3bar = (X*Y)**(-1)
    I1bar = X**2 + Y**2 + lam3bar**2 # invariant I1bar
    I2bar32 = (X**2*Y**2 + Y**2*lam3bar**2 + X**2*lam3bar**2)**(3/2) # invariant I2bar32

    Z1 = Wbar1(I1bar, I2bar32)
    if Wbar2 is None:
        pass
    elif Wbar2 == "Ogden":
        Z2 = theta2[0] * (X**theta2[1] + Y**theta2[1] + lam3bar**theta2[1] - 3)
    else:
        raise ValueError("Model not implemented yet")
    
    # plot
    fig, ax = plt.subplots()
    if Wbar2 is None:
        levels = np.linspace(np.nanmin(Z1), np.nanmax(Z1), contour_levels)
        contours = ax.contour(X, Y, Z1, levels=levels, colors=tab20_orange)
        ax.clabel(
            contours,
            inline=True,
            fontsize=8,
            fmt="%.2f" # format numbers
        )
        legend_elements = [
            Line2D([0], [0], color=tab20_orange, lw=2, label=r"$\bar W$"),
        ]
    else:
        levels1 = np.linspace(np.nanmin(Z1), np.nanmax(Z1), contour_levels)
        levels2 = np.linspace(np.nanmin(Z2), np.nanmax(Z2), contour_levels)
        contours1 = ax.contour(X, Y, Z1, levels=levels1, colors=tab20_orange)
        contours2 = ax.contour(X, Y, Z2, levels=levels2, colors=tab20_blue, linestyles='--')
        ax.clabel(
            contours1,
            inline=True,
            fontsize=8,
            fmt="%.2f" # format numbers
        )
        ax.clabel(
            contours2,
            inline=True,
            fontsize=8,
            fmt="%.2f" # format numbers
        )
        legend_elements = [
            Line2D([0], [0], color=tab20_orange, lw=2, label=r"$\bar W$ (PANO)"),
            Line2D([0], [0], color=tab20_blue, lw=2, linestyle="--", label=fr"$\bar W$ ({Wbar2})"),
        ]
    ax.legend(handles=legend_elements)
    ax.set_xlim(lam1bar_min, lam2bar_max)
    ax.set_ylim(lam1bar_min, lam2bar_max)
    plt.xlabel(r"$\bar{\lambda}_1$")
    plt.ylabel(r"$\bar{\lambda}_2$")
    ax.set_box_aspect(1)
    
    return fig, ax, legend_elements

def plot_mesh(coordinates, connectivity):
    triang = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], connectivity)
    plt.figure()
    plt.triplot(triang, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$X_1$")
    plt.ylabel(r"$X_2$")

def plot_meshes(coordinates1, connectivity1, coordinates2, connectivity2):
    triang1 = mtri.Triangulation(coordinates1[:, 0], coordinates1[:, 1], connectivity1)
    triang2 = mtri.Triangulation(coordinates2[:, 0], coordinates2[:, 1], connectivity2)
    plt.figure()
    plt.triplot(triang1, color="black", linewidth=0.5)
    plt.triplot(triang2, color="red", linewidth=0.5)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$X_1$")
    plt.ylabel(r"$X_2$")
    
def plot_scalar_field(coordinates, connectivity, scalar_field, label=None, cmap="viridis"):
    vmin = np.min(scalar_field)
    vmax = np.max(scalar_field)
    if vmax - vmin < 1e-6:
        # make sure that a homogeneous field does not look heterogeneous
        center = 0.5 * (vmin + vmax)
        eps = max(0.1 * center, 1e-6)
        vmin = center - eps
        vmax = center + eps
    triang = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], connectivity)
    plt.figure()
    plt.tripcolor(triang, scalar_field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=label)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$X_1$")
    plt.ylabel(r"$X_2$")
