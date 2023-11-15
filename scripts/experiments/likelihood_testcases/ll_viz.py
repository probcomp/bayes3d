import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL
from PIL import Image
import os
import plotly.graph_objs as go
import imageio
import io

def get_bayes3d_grid_PIL_frame(ll_per_pixel, scale=1):
    vmin = ll_per_pixel.min()
    vmax = ll_per_pixel.max()

    # Flip the data about the horizontal axis
    flipped_data = np.flip(ll_per_pixel, axis=0)  

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=flipped_data,
        zmin=vmin,  # Set the min for the color range
        zmax=vmax,  # Set the max for the color range
        # colorbar=dict(title='ll')  # Set the title for the color bar
    ))

    # Calculate aspect ratio and determine size
    aspect_ratio = ll_per_pixel.shape[0] / ll_per_pixel.shape[1]
    size = max(ll_per_pixel.shape) * scale

    # Update layout
    fig.update_layout(
        width=size if aspect_ratio < 1 else size * aspect_ratio,
        height=size if aspect_ratio > 1 else size / aspect_ratio,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    # Convert the figure to a PIL image without saving to disk
    pil_image = Image.open(io.BytesIO(fig.to_image(format='png')))
    return pil_image
