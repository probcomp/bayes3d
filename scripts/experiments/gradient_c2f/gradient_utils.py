import os
import sys
import torch 
import pytorch3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------

# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

    
#----------------------------------------------------------------------------
# Added Pose/Render Functions
#----------------------------------------------------------------------------

# Convert quaternion and position vector to 4x4 rotation matrix.
def q_v_to_mtx(q, v):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.reshape(v, (3,1))], dim=1) 
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

# Convert quaternion and position vector to 4x4 rotation matrix.
def q_v_to_mtx_batch(qs, vs):
    return torch.stack([q_v_to_mtx(q, v) for q,v in zip(qs, vs)])


# Get a random position near the origin.
def v_rnd(delta=0.005):
    x, y, z = np.random.uniform(-delta, delta, size=[3])
    return np.asarray([x, y, z], np.float32)

def q_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


#----------------------------------------------------------------------------
# Pytorch3d
#----------------------------------------------------------------------------
device = 'cuda'

def posevec_to_matrix_single(position, quat):
    return torch.cat(
        (
            torch.cat((pytorch3d.transforms.quaternion_to_matrix(quat), position.unsqueeze(1)), 1),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]],device=device),
        ),
        0,
    )

def posevec_to_matrix_batch(positions, quats):
    batch_size = positions.shape[0]
    return torch.cat(
        (
            torch.cat((pytorch3d.transforms.quaternion_to_matrix(quats), positions.unsqueeze(2)), 2),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(batch_size,1,1),
        ),
        1,
    )


#----------------------------------------------------------------------------
# Viz
#----------------------------------------------------------------------------
# import matplotlib.font_manager as fm

# fpath = os.path.join(b.utils.get_assets_dir(), "fonts","IBMPlexSerif-Regular.ttf")
# font_prop = fm.FontProperties(fname=fpath)
# font_prop

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_polar_angles_on_frame(thetas, phis, curr_ax):
    ax = curr_ax
    scaling = 0.96
    for theta in thetas:
        for phi in phis:
            x = np.cos(phi)*np.cos(theta)
            y = np.cos(phi)*np.sin(theta)
            z = np.sin(phi)
            ax.scatter(x * scaling, y, z, s=5**2, color="red", alpha=1)
            u, v, w = 1,0,0
#             ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True, alpha=0.8)
# _, ax = generate_sphere_plot()
# phis = np.arange(0, np.pi, np.pi/10)
# thetas = [0]
# plot_polar_angles_on_frame(thetas, phis, ax)

def plot_cartesian_point_on_frame(point, curr_ax, color="red", alpha=1):
    ax = curr_ax
    x, y, z = point
    ax.scatter(x, y, z, s=5**2, color=color, alpha=alpha)
    
def plot_rot_and_pos(rot_pt, pos_pt, ax_r, ax_p, color="red", alpha=1, label=None, rot_title=None, pos_title=None):
    """Given points on the spherical coord and the cartesian coord,
    Plot on the corresponding rotation and position axes"""
    rx, ry, rz = rot_pt[..., 0], rot_pt[..., 1], rot_pt[..., 2]
    px, py, pz = pos_pt[..., 0], pos_pt[..., 1], pos_pt[..., 2]
    ax_r.scatter(rx, ry, rz, s=5**2, color=color, alpha=alpha, label=label)
    ax_p.scatter(px, py, pz, s=5**2, color=color, alpha=alpha, label=label)
    
    if label is not None:
        ax_r.legend()
    if rot_title is not None:
        ax_r.set_title(rot_title)
    if pos_title is not None:
        ax_p.set_title(pos_title)
        
def plot_rot_and_pos_errors(rot_err, pos_err, x, ax_r, ax_p, color="red", alpha=1, label=None, rot_title=None, pos_title=None):
    """Given points on the spherical coord and the cartesian coord,
    Plot on the corresponding rotation and position axes"""
#     rx, ry, rz = rot_pt[..., 0], rot_pt[..., 1], rot_pt[..., 2]
#     px, py, pz = pos_pt[..., 0], pos_pt[..., 1], pos_pt[..., 2]
    ax_r.scatter(x, rot_err, s=5**2, color=color, alpha=alpha, label=label)
    ax_p.scatter(x, pos_err, s=5**2, color=color, alpha=alpha, label=label)
    
    if label is not None:
        ax_r.legend()
    if rot_title is not None:
        ax_r.set_title(rot_title)
    if pos_title is not None:
        ax_p.set_title(pos_title)

def generate_sphere_plot(show_unit=False, fig_ax=None):
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = fig_ax
        
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    u, v = np.mgrid[0:2*np.pi:21j, 0:np.pi:11j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.set_axis_off()
    ax.axes.set_xlim3d(-1.05, 1.05) 
    ax.axes.set_ylim3d(-1.05, 1.05) 
    ax.axes.set_zlim3d(-1.05, 1.05) 
    ax.set_aspect("equal")
    ax.plot_wireframe(x, y, z, color=(0.0, 0.0, 0.0, 0.3), linewidths=0.5)

    ax.axes.set_xlabel("x")
    ax.axes.set_ylabel("y")
    ax.axes.set_zlabel("z")

    if show_unit:
        quat_unit = q_to_mtx(torch.tensor([1,0,0,0], device="cuda", dtype=torch.float64)).cpu()[:3, :3] @ torch.tensor(UNIT_VECTOR, dtype=torch.float64) 
        ax.scatter(quat_unit[0], quat_unit[1], quat_unit[2], color="green", alpha=1)
    return fig, ax

def generate_cartesian_plot(show_unit=False, fig_ax=None):
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = fig_ax
    
    ax.axes.set_xlim3d(-0.5, 0.5) 
    ax.axes.set_ylim3d(-0.5, 0.5) 
    ax.axes.set_zlim3d(-1.2, 1.2) 
    
    ax.axes.set_xlabel("x")
    ax.axes.set_ylabel("y")
    ax.axes.set_zlabel("z")
    
    if show_unit:
        ax.scatter(0.0, 0.0, 0.0, s=5**2, color="green", alpha=1)
    return fig, ax

def generate_rotation_translation_plot(show_unit=False):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.33))

    # set up the axes for the plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Generate the subplots
    _, _ = generate_sphere_plot(show_unit, (fig, ax1))
    _, _ = generate_cartesian_plot(show_unit, (fig, ax2))

    # Label with title
    ax1.set_title("Rotation evolution")
    ax2.set_title("Translation evolution")
    
    return fig, (ax1, ax2)

def generate_rotation_translation_err_plot(T):
    fig = plt.figure(figsize=plt.figaspect(0.2))
    fpath = os.path.join(b.utils.get_assets_dir(), "fonts","IBMPlexSerif-Regular.ttf")
    font_prop = fm.FontProperties(fname=fpath)
    
    # set up the axes 
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    # axis limits and formatting
    stepsize = max(1, T // 10)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(left=0, right=T)
        ax.set_xticks(range(0,T+1,stepsize))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # axis labeling + stylization
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Iterations", font=font_prop, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_prop)        

    # Title
    ax1.set_title("Rotation error", font=font_prop, fontsize=15)
    ax2.set_title("Translation error",font=font_prop, fontsize=15)
    ax3.set_title("Loss",font=font_prop, fontsize=15)
        
    return fig, (ax1, ax2, ax3)

def target_reconstruction_err_plot(T):
    fig = plt.figure(figsize=(6,6))
    fpath = os.path.join(b.utils.get_assets_dir(), "fonts","IBMPlexSerif-Regular.ttf")
    font_prop = fm.FontProperties(fname=fpath)
    
    # set up the axes 
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,1,2)
    
    # axis limits and formatting
    stepsize = max(1, T // 10)
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    for ax in [ax3]:
        ax.set_xlim(left=0, right=T)
        ax.set_xticks(range(0,T+1,stepsize))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # axis labeling + stylization
    for ax in [ax3]:
        ax.set_xlabel("Iterations", font=font_prop, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_prop)          
    
    ax1.set_title("Target",font=font_prop, fontsize=15)
    ax2.set_title("Best Reconstruction",font=font_prop, fontsize=15)
    ax3.set_title("Pixelwise MSE Loss",font=font_prop, fontsize=15)
    
    fig.tight_layout()    
    return fig, (ax1, ax2, ax3)

def get_img_with_border(img, border=5, fill='red'):
    cropped_img = ImageOps.crop(img, border=border)
    return ImageOps.expand(cropped_img, border=border,fill=fill)


#----------------------------------------------------------------------------
# Reproducibility
#----------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# ---------------------------------------------------------------------------
# GRADIENT DESCENT
# ---------------------------------------------------------------------------
def descend_gradient(pose_rot_opt, pose_pos_opt, pose_target, rast_opt, rast_target, it=20, verbose=False, plot=False):
    OPTIM_GIF_IMGS = []
    loss_best = np.inf
    optimizer = torch.optim.Adam([pose_rot_opt, pose_pos_opt], betas=(0.9, 0.999), lr=lr_base)
    
    if plot: 
        fig, (ax_r, ax_t) = generate_rotation_translation_plot(False)
        plot_rot_and_pos(pose_opt.detach().cpu()[:3, :3] @ UNIT_VECTOR, 
                 pose_opt.detach().cpu()[:3, -1], 
                 ax_r, ax_t, 
                 color="green", alpha=1, label="Initial")
        plot_rot_and_pos(pose_opt.detach().cpu()[:3, :3] @ UNIT_VECTOR, 
                         pose_opt.detach().cpu()[:3, -1], 
                         ax_r, ax_t, 
                         color="blue", alpha=0.1, label="Hypothesis")
        plot_rot_and_pos(pose_target.detach().cpu()[:3, :3] @ UNIT_VECTOR, 
                         pose_target.detach().cpu()[:3, -1], 
                         ax_r, ax_t, 
                         color="red", alpha=1, label="Target")

        
    for i in tqdm(range(it)):
        noise = q_unit()
        pose_rot_total_opt = q_mul_torch(pose_rot_opt, noise)
        mtx_total_opt  = torch.matmul(mvp, q_v_to_mtx(pose_rot_total_opt, pose_pos_opt))
        color_opt      = render(glctx, mtx_total_opt, vtx_pos, pos_idx, vtx_col, col_idx, resolution)

        diff = (rast_opt - rast_target)**2 # L2 norm.
        diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
        loss = torch.mean(diff)
        loss_val = float(loss)

        if (loss_val < loss_best) and (loss_val > 0.0):
            loss_best = loss_val
        if (loss_val/loss_best > 1.2):
            break
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pose_rot_opt /= torch.sum(pose_rot_opt**2)**0.5

        rast_opt = render(glctx, torch.matmul(mvp, q_v_to_mtx(pose_rot_opt, pose_pos_opt)), vtx_pos, pos_idx, vtx_col, col_idx, resolution)
        img_opt  = rast_opt[0].detach().cpu().numpy()
    
    
        if verbose:
            print(f"loss={loss}, rot={pose_rot_total_opt}, pos={pose_pos_opt}")

        curr_render_imgs = b.hstack_images([
                b.get_depth_image(img_opt[:,:,0]* 255.0) ,
                b.get_depth_image(img_target[:,:,0]* 255.0) ,
                ])
        
        if plot:
            pose_opt_curr_val = q_v_to_mtx(pose_rot_opt, pose_pos_opt).detach().cpu() 
            plot_rot_and_pos(pose_opt_curr_val[:3, :3] @ UNIT_VECTOR, 
                         pose_opt_curr_val[:3, -1], 
                         ax_r, ax_t, 
                         color="blue", alpha=0.1,
                         rot_title=f"Rotation evolution, iter {i}", 
                         pos_title=f"Translation evolution, iter {i}")  # current
            curr_PIL = fig2img(fig)
            OPTIM_GIF_IMGS.append(b.hstack_images([curr_PIL, curr_render_imgs]))
        else:
            OPTIM_GIF_IMGS.append(curr_render_imgs)

    return OPTIM_GIF_IMGS


def descend_gradient_multi(pose_rot_opt, pose_pos_opt, pose_target, rast_opts, rast_target, it=20, verbose=False, plot=False):
    OPTIM_GIF_IMGS = []
    loss_best = np.inf
    optimizer = torch.optim.Adam([pose_rot_opt, pose_pos_opt], betas=(0.9, 0.999), lr=2e-7)
    img_target = rast_target[0].detach().cpu().numpy()
    img_target_viz = b.get_depth_image(img_target[:,:,0]* 255.0)
    
    if plot: 
        fig, (ax_r, ax_t) = generate_rotation_translation_plot()
        
        poses_opt = q_v_to_mtx_batch(poses_rot_opt, poses_pos_opt)

        plot_rot_and_pos(np.einsum('nij,j... -> ni', poses_opt.detach().cpu()[:, :3, :3], UNIT_VECTOR), 
                         poses_opt.detach().cpu()[:, :3, -1], 
                         ax_r, ax_t, 
                         color="green", alpha=0.1, label="Initial")
        plot_rot_and_pos(np.einsum('nij,j... -> ni', poses_opt.detach().cpu()[:, :3, :3], UNIT_VECTOR),
                         poses_opt.detach().cpu()[:, :3, -1], 
                         ax_r, ax_t, 
                         color="blue", alpha=0.1, label="Hypothesis")
        plot_rot_and_pos(pose_target.detach().cpu()[:3, :3] @ UNIT_VECTOR, 
                         pose_target.detach().cpu()[:3, -1], 
                         ax_r, ax_t, 
                         color="red", alpha=1, label="Target")

    
    # TODO better convergence condition
    for i in tqdm(range(it)):
    #     noise = q_unit()
        poses_rot_total_opt = poses_rot_opt #q_mul_torch(pose_rot_opt, noise)
        mtx_total_opt  = torch.matmul(mvp, q_v_to_mtx_batch(poses_rot_total_opt, poses_pos_opt))
        color_opts = render_multiple(glctx, mtx_total_opt, vtx_pos, pos_idx, vtx_col, col_idx, resolution)

        diff = (rast_opts - rast_target)**2 # L2 norm.
        diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
        loss = torch.mean(diff)
        loss_val = float(loss)

        if (loss_val < loss_best) and (loss_val > 0.0):
            loss_best = loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #     with torch.no_grad():
    #         pose_rot_opt /= torch.sum(poses_rot_opt**2, axis=1)**0.5

        rast_opts = render_multiple(glctx, torch.matmul(mvp, q_v_to_mtx_batch(poses_rot_opt, poses_pos_opt)), vtx_pos, pos_idx, vtx_col, col_idx, resolution)
        img_opts  = rast_opts.detach().cpu().numpy()
    
        if verbose:
            print(f"loss={loss}, pos[0]={pose_pos_opt[0]}")

        curr_render_imgs = b.hvstack_images([b.get_depth_image(img_opts[i][:,:,0]* 255.0) for i in range(len(rast_opts))], 
                                       fibonacci_sphere_points*num_x,
                                       num_planar_angle_points*num_y*num_z,
                                       border=10)
        curr_render_imgs = b.vstack_images([img_target_viz, b.scale_image(curr_render_imgs, 0.3)])

        if plot:
            poses_opt_curr_val = q_v_to_mtx_batch(poses_rot_opt, poses_pos_opt).detach().cpu() 
            plot_rot_and_pos(np.einsum('nij,j... -> ni', poses_opt_curr_val[:, :3, :3], UNIT_VECTOR), 
                         poses_opt_curr_val[:, :3, -1], 
                         ax_r, ax_t, 
                         color="blue", alpha=0.1,
                         rot_title=f"Rotation evolution, iter {i}", 
                         pos_title=f"Translation evolution, iter {i}")  # current
            curr_fig = fig2img(fig)
            OPTIM_GIF_IMGS.append(b.hstack_images([curr_fig, curr_render_imgs]))
        else:
            OPTIM_GIF_IMGS.append(curr_render_imgs)

    return OPTIM_GIF_IMGS