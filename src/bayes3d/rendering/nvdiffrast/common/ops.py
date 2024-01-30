# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch

import bayes3d.rendering.nvdiffrast.nvdiffrast_plugin_gl as plugin_gl

# ----------------------------------------------------------------------------
# C++/Cuda plugin loader.


def _get_plugin(gl=True):
    # the gl flag is left here for backward compatibility
    assert gl is True
    return plugin_gl


# ----------------------------------------------------------------------------
# GL state wrapper.
# ----------------------------------------------------------------------------


class RasterizeGLContext:
    def __init__(self, height, width, output_db=False, mode="automatic", device=None):
        """Create a new OpenGL rasterizer context.

        Creating an OpenGL context is a slow operation so you should usually reuse the same
        context in all calls to `rasterize()` on the same CPU thread. The OpenGL context
        is deleted when the object is destroyed.

        Side note: When using the OpenGL context in a rasterization operation, the
        context's internal framebuffer object is automatically enlarged to accommodate the
        rasterization operation's output shape, but it is never shrunk in size until the
        context is destroyed. Thus, if you need to rasterize, say, deep low-resolution
        tensors and also shallow high-resolution tensors, you can conserve GPU memory by
        creating two separate OpenGL contexts for these tasks. In this scenario, using the
        same OpenGL context for both tasks would end up reserving GPU memory for a deep,
        high-resolution output tensor.

        Args:
          output_db (bool): Compute and output image-space derivates of barycentrics.
          mode: OpenGL context handling mode. Valid values are 'manual' and 'automatic'.
          device (Optional): Cuda device on which the context is created. Type can be
                             `torch.device`, string (e.g., `'cuda:1'`), or int. If not
                             specified, context will be created on currently active Cuda
                             device.
        Returns:
          The newly created OpenGL rasterizer context.
        """
        assert output_db is True or output_db is False
        assert mode in ["automatic", "manual"]
        self.output_db = output_db
        self.mode = mode
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cpp_wrapper = _get_plugin(gl=True).RasterizeGLStateWrapper(
            output_db, mode == "automatic", cuda_device_idx
        )
        self.active_depth_peeler = None  # For error checking only.

    def set_context(self):
        """Set (activate) OpenGL context in the current CPU thread.
        Only available if context was created in manual mode.
        """
        assert self.mode == "manual"
        self.cpp_wrapper.set_context()

    def release_context(self):
        """Release (deactivate) currently active OpenGL context.
        Only available if context was created in manual mode.
        """
        assert self.mode == "manual"
        self.cpp_wrapper.release_context()
