import os
import warnings

import setuptools
from torch.utils import cpp_extension

CPP_SRC_DIR = "src/bayes3d/rendering/nvdiffrast/common"

# Nvdiffrast compilation setup
if os.name == "nt":
    lib_dir = os.path.dirname(__file__) + r"\..\lib"

    def find_cl_path():
        import glob

        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            vs_relative_path = (
                r"\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64"
                % edition
            )
            paths = sorted(
                glob.glob(r"C:\Program Files" + vs_relative_path), reverse=True
            )
            paths += sorted(
                glob.glob(r"C:\Program Files (x86)" + vs_relative_path), reverse=True
            )
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError(
                "Could not locate a supported Microsoft Visual C++ installation"
            )
        os.environ["PATH"] += ";" + cl_path

# Compiler options.
opts = ["-DNVDR_TORCH"]

# Linker options for the GL-interfacing plugin.
ldflags = []
if os.name == "posix":
    ldflags = ["-lGL", "-lEGL"]
elif os.name == "nt":
    libs = ["gdi32", "opengl32", "user32", "setgpu"]
    ldflags = ["/LIBPATH:" + lib_dir] + ["/DEFAULTLIB:" + x for x in libs]

# List of source files.
source_files = [
    "common.cpp",
    "glutil.cpp",
    "rasterize_gl.cpp",
]
source_files = [os.path.join(CPP_SRC_DIR, fn) for fn in source_files]

# Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# On Linux, show a warning if GLEW is being forcibly loaded when compiling the GL plugin.
if (os.name == "posix") and ("libGLEW" in os.environ.get("LD_PRELOAD", "")):
    warnings.warn(
        "libGLEW is being loaded via LD_PRELOAD, and will probably conflict with the OpenGL plugin"
    )

# Speed up compilation on Windows.
if os.name == "nt":
    # Skip telemetry sending step in vcvarsall.bat
    os.environ["VSCMD_SKIP_SENDTELEMETRY"] = "1"

    # Opportunistically patch distutils to cache MSVC environments.
    try:
        import distutils._msvccompiler
        import functools

        if not hasattr(distutils._msvccompiler._get_vc_env, "__wrapped__"):
            distutils._msvccompiler._get_vc_env = functools.lru_cache()(
                distutils._msvccompiler._get_vc_env
            )
    except Exception:
        pass

setuptools.setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="bayes3d.rendering.nvdiffrast.nvdiffrast_plugin_gl",
            sources=source_files,
            extra_compile_args={"cxx": opts, "nvcc": opts + ["-lineinfo"]},
            extra_link_args=ldflags,
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
