import gc
import os

import bayes3d as b
import jax.numpy as jnp

# setup renderer
intrinsics = b.Intrinsics(50, 50, 200.0, 200.0, 25.0, 25.0, 0.001, 10.0)
# Note: removing the b.RENDERER object does the same operation in C++ as clear_meshmem()
b.setup_renderer(intrinsics, num_layers=1)
renderer = b.RENDERER

pre_test_clearmesh = b.utils.get_gpu_memory()[0]

for i in range(5):
    b.setup_renderer(intrinsics, num_layers=1)
    renderer = b.RENDERER

    pre_add_mesh = b.utils.get_gpu_memory()[0]
    for x in range(1):
        renderer.add_mesh_from_file(
            os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"),
            mesh_name=f"cube_{i+1}",
        )

    post_add_mesh = b.utils.get_gpu_memory()[0]

    pose = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    depth = renderer.render(pose[None, ...], jnp.array([0]))[..., 2]

    post_render = b.utils.get_gpu_memory()[0]

    renderer.clear_gpu_meshmem()

    post_clear_meshmem = b.utils.get_gpu_memory()[0]

    # ensure the mesh memory is fully cleared
    assert pre_add_mesh - post_add_mesh == post_clear_meshmem - post_render

    gc.collect()

    print(f"{i}: ", b.utils.get_gpu_memory()[0])

post_test_clearmesh = b.utils.get_gpu_memory()[0]

# Expected result should be around 2MiB for the given camera intrinsics
print(
    "GPU memory lost with clear_meshmem() --> ",
    pre_test_clearmesh - post_test_clearmesh,
    " MiB",
)
print(
    "The memeory lost is from the JAX memeory in GPU and not accumulations in the GPU"
)
