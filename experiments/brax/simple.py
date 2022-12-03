from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import brax
from IPython.display import HTML, Image 

import time
from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
from brax.io import mesh
import jax
from jax import numpy as jnp

#@title A bouncy ball scene
bouncy_ball = brax.Config(dt=0.05, substeps=20, dynamics_mode='pbd')

# ground is a frozen (immovable) infinite plane
ground = bouncy_ball.bodies.add(name='ground')
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()  # for setting an empty oneof

# ball weighs 1kg, has equal rotational inertia along all axes, is 1m long, and
# has an initial rotation of identity (w=1,x=0,y=0,z=0) quaternion
ball = bouncy_ball.bodies.add(name='ball', mass=1)
cap = ball.colliders.add().capsule
cap.radius, cap.length = 0.5, 1

# gravity is -9.8 m/s^2 in z dimension
bouncy_ball.gravity.z = -9.8

qp = brax.QP(
    # position of each body in 3d (z is up, right-hand coordinates)
    pos = np.array([[0., 0., 0.],       # ground
                    [0., 0., 3.]]),     # ball is 3m up in the air
    # velocity of each body in 3d
    vel = np.array([[0., 0., 0.],       # ground
                    [0., 0., 0.]]),     # ball
    # rotation about center of body, as a quaternion (w, x, y, z)
    rot = np.array([[1., 0., 0., 0.],   # ground
                    [1., 0., 0., 0.]]), # ball
    # angular velocity about center of body in 3d
    ang = np.array([[0., 0., 0.],       # ground
                    [0., 0., 0.]])      # ball
)


#@title Simulating the bouncy ball config { run: "auto"}
bouncy_ball.elasticity = 0.85 #@param { type:"slider", min: 0, max: 1.0, step:0.05 }
ball_velocity = 1 #@param { type:"slider", min:-5, max:5, step: 0.5 }

sys = brax.System(bouncy_ball)

# provide an initial velocity to the ball
qp.vel[1, 0] = ball_velocity

stepper_jit = jax.jit(sys.step)
qp, _ = stepper_jit(qp, [])

num_timesteps = 100
start = time.time()
for i in range(num_timesteps):
    qp, _ = stepper_jit(qp, [])
end = time.time()

print('FPS:');print(num_timesteps / (end-start))

from IPython import embed; embed()