import math
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_img(t):
    t = t.detach().cpu().squeeze(0).numpy()
    fig = px.imshow(t, color_continuous_scale="gray")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.show()

def display_imgs(t1, t2):
    t1 = t1.detach().cpu().squeeze(0).numpy()
    t2 = t2.detach().cpu().squeeze(0).numpy()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02)
    fig.add_trace(
        go.Heatmap(z=t1, colorscale="gray", showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=t2, colorscale="gray", showscale=False),
        row=1, col=2
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()

def plot_images(images, labels, preds, ncols=8):
    """
    images: torch.Tensor of shape [N, 1, 28, 28]
    ncols: number of columns
    """
    N = images.shape[0]
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(N):
        img = images[i, 0].detach().cpu()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {labels[i]}\nPred: {preds[i]}")
        axes[i].axis("off")

    # Hide unused cells
    for i in range(N, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

def plot_image_matches(images, labels, preds, train_matches_x, train_matches_y):
    k = train_matches_x.shape[1]
    nrows = len(images)
    ncols = k + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    if nrows == 1:
        axes = [axes]

    for i in range(nrows):
        row_axes = axes[i]
        img = images[i, 0].detach().cpu()
        row_axes[0].imshow(img, cmap="gray")
        row_axes[0].set_title(f"Label: {labels[i]}\nPred: {preds[i]}")
        row_axes[0].axis("off")

        for j in range(k):
            match_img = train_matches_x[i, j, 0].detach().cpu()
            match_y = train_matches_y[i, j]
            row_axes[j + 1].imshow(match_img, cmap="gray")
            row_axes[j + 1].set_title(f"Train: {match_y}")
            row_axes[j + 1].axis("off")

    plt.tight_layout()
    plt.show()
