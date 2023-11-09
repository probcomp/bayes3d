import numpy as np
import bayes3d as b
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL
import os

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
                b.viz.scale_image(b.viz.get_depth_image(rendered[i,...,2]), scale),
                b.viz.scale_image(b.viz.get_depth_image(observed[i,...,2]), scale)
            ],
            labels=["Inferred", "Observed"],
            label_fontsize=20
        )
        for i in range(observed.shape[0])
    ]
    return display_video(images, framerate=framerate)

def plot_bayes3d_likelihood(gt_images, rendered, fps = 5, ll_f = b.threedp3_likelihood_per_pixel_old, ll_f_args = (0.0001,0.0001,1000,3)):

    T = gt_images.shape[0]

    data = np.stack([ll_f(gt_images[i], rendered[i],*ll_f_args) for i in range(T)])
    # Determine the consistent color range
    vmin = data.min()
    vmax = data.max()

    # Create a directory to store the frames
    if not os.path.exists("frames"):
        os.makedirs("frames")

    # Generate and save each flipped frame
    for t in range(data.shape[0]):
        flipped_data = np.flip(data[t], axis=0)  # Flip the data about the horizontal axis
        fig = go.Figure(data=go.Heatmap(
            z=flipped_data,
            zmin=vmin,  # Set the min for the color range
            zmax=vmax,  # Set the max for the color range
            colorbar=dict(title='Bayes3D Likelihood')  # Set the title for the color bar
        ))

        # Update layout to ensure the same aspect ratio for both axes
        fig.update_layout(
            width=700, 
            height=500, 
            autosize=True,
            margin=dict(l=10, r=10, b=10, t=10),
            xaxis=dict(
                scaleratio=1
            ),
            yaxis=dict(
                scaleratio=1
            )
        )
        
        # Save as an image
        frame_filename = f"frames/heatmap_{t:03d}.png"
        fig.write_image(frame_filename)

    # Create the GIF using imageio
    images = []
    for t in range(data.shape[0]):
        frame_filename = f"frames/heatmap_{t:03d}.png"
        images.append(imageio.imread(frame_filename))

    # Save to GIF
    imageio.mimsave('heatmap_animation.gif', images, fps=fps)  # Set fps as needed

    # Optionally, remove the frames directory if you no longer need it
    import shutil
    shutil.rmtree("frames")




def plot_3d_poses(poses_list, poses_names = ["Ground Truth", "Inferred"], name = "default", fps = 10, save = False):
    """
    poses_list is a list of T x 4 x 4 poses
    pose_names is a list of names, same len as poses_list
    """

    assert len(poses_list) == len(poses_names)

    num_paths = len(poses_list)
    T = poses_list[0].shape[0]

    def get_walk(poses_list):
        walks = np.zeros((num_paths,T,3))
        for i in range(num_paths):
            for j in range(T):
                walks[i,j,:] = poses_list[i][j,:3,3]
        return walks
    
    walks = get_walk(poses_list)

    # Generate unique colors for each walk
    colors = ['rgba(255,0,0,0.8)']
    if num_paths == 2:
        colors = colors + ['rgba(0,0,255,0.8)']
    elif num_paths > 2:
     colors = colors + [f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)' for _ in range(num_paths - 1)]

    # Find the axis ranges based on the walks
    all_walks = walks.reshape(-1, 3)  # Reshape for simplicity
    x_range = [all_walks[:, 0].min(), all_walks[:, 0].max()]
    y_range = [all_walks[:, 1].min(), all_walks[:, 1].max()]
    z_range = [all_walks[:, 2].min(), all_walks[:, 2].max()]

    y_asp = (y_range[1]-y_range[0])/(x_range[1]-x_range[0])
    z_asp = (z_range[1]-z_range[0])/(x_range[1]-x_range[0])

    # Function to generate a random rotation matrix
    def random_rotation_matrix():
        random_rotation = R.random()
        return random_rotation.as_matrix()

    # Define a function to create an arrow at a given point and direction
    def create_arrow(point, direction, color='red', length_scale=1, showlegend=False):
        length = length_scale * np.linalg.norm([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]) * 0.1
        # Normalize the direction
        direction = direction / np.linalg.norm(direction)
        
        # Create the arrow components (shaft and head)
        shaft = go.Scatter3d(
            x=[point[0], point[0] + direction[0] * length],
            y=[point[1], point[1] + direction[1] * length],
            z=[point[2], point[2] + direction[2] * length],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=showlegend
        )
        
        # head = go.Cone(
        #     x=[point[0] + direction[0] * length],
        #     y=[point[1] + direction[1] * length],
        #     z=[point[2] + direction[2] * length],
        #     u=[direction[0]],
        #     v=[direction[1]],
        #     w=[direction[2]],
        #     sizemode='absolute',
        #     sizeref=0.001,
        #     anchor='tip',
        #     colorscale=[[0, color], [1, color]],
        #     showscale=False,
        #     showlegend=showlegend
        # )
        
        # return shaft, head
        return shaft, None

    # Create a directory for frames
    if not os.path.exists("frames"):
        os.mkdir("frames")

    # Generate each frame and save as an image file
    image_files = []

    # Generating frames and saving as image files, the loop for this
    for i in range(T):
        frame_data = []
        for w in range(num_paths):
            rotation_matrix = poses_list[w][T,:3,:3]

            eigenvalues, eigenvectors = np.linalg.eig(rotation_matrix)

            # The axis of rotation is the eigenvector corresponding to the eigenvalue of 1
            axis_of_rotation = eigenvectors[:, np.isclose(eigenvalues, 1)]

            # Make sure it's a unit vector
            axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)
            arrow_direction = np.real(axis_of_rotation)

            # arrow_direction = rotation_matrix @ np.array([1, 0, 0])
            
            # Pass showlegend as False to the create_arrow function
            shaft, head = create_arrow(walks[w, i], arrow_direction, color=colors[w], showlegend=False)
            
            # Add name and legendgroup to the Scatter3d trace
            trace = go.Scatter3d(
                x=walks[w, :i+1, 0],
                y=walks[w, :i+1, 1],
                z=walks[w, :i+1, 2],
                mode='markers+lines',
                marker=dict(size=5, color=colors[w]),
                line=dict(color=colors[w], width=2),
                name=poses_names[w],  # Name for legend
                legendgroup=poses_names[w],  # Same legendgroup for walk dots and arrows
            )
            # Add only for the first frame to avoid duplicate legend entries
            if i == 0:
                trace.legendgrouptitle = dict(text=poses_names[w])

            frame_data.extend([trace, shaft])
        
        # Define the figure for the current frame
        fig = go.Figure(
            data=frame_data,
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(range=x_range, autorange=False),
                    yaxis=dict(range=y_range, autorange=False),
                    zaxis=dict(range=z_range, autorange=False),
                    # aspectratio=dict(x=1, y=y_asp, z=z_asp),
                    aspectratio=dict(x=1, y=1, z=1),
                    camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                )
                ),
                margin=dict(l=0, r=0, t=0, b=0)  # Reduce white space around the plot
            )
        )
        if save:
            # Save the figure as an image file
            img_file = f'frames/frame_{i:03d}.png'
            fig.write_image(img_file)
            image_files.append(img_file)

    if not save:
        fig.show()
    else:     
        # # Create a GIF using the saved image files
        with imageio.get_writer(f'{name}.gif', mode='I', fps=fps, loop = 0) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
                # Optionally, remove the image file after adding it to the GIF
                os.remove(filename)  

        # Clean up the frames directory if desired
        os.rmdir("frames")

        print(f"GIF saved as '{name}.gif'")
