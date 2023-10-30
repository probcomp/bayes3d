import numpy as np
import bayes3d as b
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL

def display_video(frames, framerate=30):
    """
    frames: PIL Image OR a list of N np.arrays (H x W x 3)
    framerate: frames per second
    """
    if type(frames[0]) == PIL.Image.Image:
      frames = [np.array(frames[i]) for i in range(len(frames))]
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=True)
    return HTML(anim.to_html5_video())

def video_from_trace(trace, scale = 8, compare = False, use_retval = False, framerate = 30, rendered_addr = ("depths", "depths")):
    if use_retval:
        rendered = trace.get_retval()[0]
    if compare:
        return video_comparison_from_trace(trace, scale, framerate, rendered_addr)
    else:
        # check if this is the address structure we plan to use
        rendered = trace[rendered_addr]
    return video_from_rendered(rendered, scale, framerate)

def video_from_rendered(rendered, scale = 8, framerate = 30):
    images = [b.scale_image(b.get_depth_image(rendered[i,...,2]),scale) for i in range(rendered.shape[0])]
    return display_video(images, framerate=framerate)

def video_comparison_from_trace(trace, scale = 8, framerate = 30, rendered_addr = ("depths", "depths")):
    rendered = trace.get_retval()[0]
    observed = trace[rendered_addr]
    return video_comparison_from_images(rendered, observed, scale, framerate)

def video_comparison_from_images(rendered, observed, scale = 8, framerate = 30):

    images = [
        b.viz.multi_panel(
            [
                b.viz.scale_image(b.viz.get_depth_image(rendered[i,...,2]), scale)
                b.viz.scale_image(b.viz.get_depth_image(observed[i,...,2]), scale),
            ],
            labels=["Inferred", "Observed"],
            label_fontsize=20
        )
        for i in range(observed.shape[0])
    ]
    return display_video(images, framerate=framerate)
