# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import importlib
import logging
import numpy as np
import os
import torch
import torch.utils.cpp_extension

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = {}
def _get_plugin(gl=False):
    assert isinstance(gl, bool)

    # Return cached plugin if already loaded.
    if _cached_plugin.get(gl, None) is not None:
        return _cached_plugin[gl]

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        lib_dir = os.path.dirname(__file__) + r"\..\lib"
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                vs_relative_path = r"\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition
                paths = sorted(glob.glob(r"C:\Program Files" + vs_relative_path), reverse=True)
                paths += sorted(glob.glob(r"C:\Program Files (x86)" + vs_relative_path), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options for the GL-interfacing plugin.
    ldflags = []
    if gl:
        if os.name == 'posix':
            ldflags = ['-lGL', '-lEGL']
        elif os.name == 'nt':
            libs = ['gdi32', 'opengl32', 'user32', 'setgpu']
            ldflags = ['/LIBPATH:' + lib_dir] + ['/DEFAULTLIB:' + x for x in libs]

    # List of source files.
    if gl:
        source_files = [
            'common.cpp',
            'glutil.cpp',
            'rasterize_gl.cpp',
            'likelihood.cu',
        ]
    else:
        source_files = [
            '../common/cudaraster/impl/Buffer.cpp',
            '../common/cudaraster/impl/CudaRaster.cpp',
            '../common/cudaraster/impl/RasterImpl.cu',
            '../common/cudaraster/impl/RasterImpl.cpp',
            '../common/common.cpp',
            '../common/rasterize.cu',
            '../common/interpolate.cu',
            '../common/texture.cu',
            '../common/texture.cpp',
            '../common/antialias.cu',
            'torch_bindings.cpp',
            'torch_rasterize.cpp',
            'torch_interpolate.cpp',
            'torch_texture.cpp',
            'torch_antialias.cpp',
        ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # On Linux, show a warning if GLEW is being forcibly loaded when compiling the GL plugin.
    if gl and (os.name == 'posix') and ('libGLEW' in os.environ.get('LD_PRELOAD', '')):
        logging.getLogger('nvdiffrast').warning("Warning: libGLEW is being loaded via LD_PRELOAD, and will probably conflict with the OpenGL plugin")

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    plugin_name = 'nvdiffrast_plugin' + ('_gl' if gl else '')
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory(plugin_name, False), 'lock')
        if os.path.exists(lock_fn):
            logging.getLogger('nvdiffrast').warning("Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Speed up compilation on Windows.
    if os.name == 'nt':
        # Skip telemetry sending step in vcvarsall.bat
        os.environ['VSCMD_SKIP_SENDTELEMETRY'] = '1'

        # Opportunistically patch distutils to cache MSVC environments.
        try:
            import distutils._msvccompiler
            import functools
            if not hasattr(distutils._msvccompiler._get_vc_env, '__wrapped__'):
                distutils._msvccompiler._get_vc_env = functools.lru_cache()(distutils._msvccompiler._get_vc_env)
        except:
            pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name=plugin_name, sources=source_paths, extra_cflags=opts, extra_cuda_cflags=opts+['-lineinfo'], extra_ldflags=ldflags, with_cuda=True, verbose=False)

    # Import, cache, and return the compiled module.
    _cached_plugin[gl] = importlib.import_module(plugin_name)
    return _cached_plugin[gl]

#----------------------------------------------------------------------------
# Log level.
#----------------------------------------------------------------------------

def get_log_level():
    '''Get current log level.

    Returns:
      Current log level in nvdiffrast. See `set_log_level()` for possible values.
    '''
    return _get_plugin().get_log_level()

def set_log_level(level):
    '''Set log level.

    Log levels follow the convention on the C++ side of Torch:
      0 = Info,
      1 = Warning,
      2 = Error,
      3 = Fatal.
    The default log level is 1.

    Args:
      level: New log level as integer. Internal nvdiffrast messages of this 
             severity or higher will be printed, while messages of lower
             severity will be silent.
    '''
    _get_plugin().set_log_level(level)

#----------------------------------------------------------------------------
# GL state wrapper.
#----------------------------------------------------------------------------

class RasterizeGLContext:
    def __init__(self, height, width, output_db=False, mode='automatic', device=None):
        '''Create a new OpenGL rasterizer context.

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
        '''
        assert output_db is True or output_db is False
        assert mode in ['automatic', 'manual']
        self.output_db = output_db
        self.mode = mode
        if device is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(device):
                cuda_device_idx = torch.cuda.current_device()
        self.cpp_wrapper = _get_plugin(gl=True).RasterizeGLStateWrapper(output_db, mode == 'automatic', cuda_device_idx)
        self.active_depth_peeler = None # For error checking only.

    def set_context(self):
        '''Set (activate) OpenGL context in the current CPU thread.
           Only available if context was created in manual mode.
        '''
        assert self.mode == 'manual'
        self.cpp_wrapper.set_context()

    def release_context(self):
        '''Release (deactivate) currently active OpenGL context.
           Only available if context was created in manual mode.
        '''
        assert self.mode == 'manual'
        self.cpp_wrapper.release_context()
