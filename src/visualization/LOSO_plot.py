from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib

# Generate the class/group data
n_points = 100
X = np.random.randn(100, 10)

percentiles_classes = [0.1, 0.3, 0.6]
y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])

# Evenly spaced groups repeated once
groups = np.hstack([[ii] * 10 for ii in range(10)])
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

col = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.2)

rgba = cmap_cv(col(1))
print(rgba)

n_splits = 4
percentiles_classes = [0.1, 0.3, 0.6]
y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])


def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(
        range(len(groups)),
        [0.5] * len(groups),
        c=groups,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(groups)),
        [3.5] * len(groups),
        c=classes,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.set(
        ylim=[-1, 5],
        yticks=[0.5, 3.5],
        yticklabels=["Data\ngroup", "Data\nclass"],
        xlabel="Sample index",
    )


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        if ii in [0,1]:
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=lw,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )
        elif ii == 9:
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii-6 + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=lw,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii-6 + 1.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    #x.scatter(
    #    range(len(X)), [ii-6 + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    #)

    # Formatting
    yticklabels = [0,1,'', 9] + ["subject"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("Leave One Subject Out", fontsize=15)
    return ax

cv = LeaveOneGroupOut()
fig, ax = plt.subplots(figsize=(6, 3))
plot_cv_indices(cv, X, y, groups, ax, n_splits)
ax.legend(
    [Patch(color=cmap_cv(col(1))), Patch(color=cmap_cv(col(0)))],
    ["Test set", "Train set"],
    loc=(1.02, 0.8),
)
# Make the legend fit
plt.tight_layout()
plt.show()

# %%
# Let's see how it looks for the :class:`~sklearn.model_selection.KFold`
# cross-validation object:

fig, ax = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)

visualize_groups(y, groups, "no groups")
plt.show()
