import setuptools
import subprocess
import os

setuptools.setup(
    name='jax3dp3',
    version='1.0.0',
    package_data={
        'jax3dp3': [
            'nvdiffrast/common/*.h',
            'nvdiffrast/common/*.inl',
            'nvdiffrast/common/*.cu',
            'nvdiffrast/common/*.cpp',
            'nvdiffrast/common/cudaraster/*.hpp',
            'nvdiffrast/common/cudaraster/impl/*.cpp',
            'nvdiffrast/common/cudaraster/impl/*.hpp',
            'nvdiffrast/common/cudaraster/impl/*.inl',
            'nvdiffrast/common/cudaraster/impl/*.cu',
            'nvdiffrast/lib/*.h',
            'nvdiffrast/torch/*.h',
            'nvdiffrast/torch/*.inl',
            'nvdiffrast/torch/*.cpp',
            'nvdiffrast/torch/*.cu',
        ] + (['nvdiffrast/lib/*.lib'] if os.name == 'nt' else [])
    },
    include_package_data=True,
    install_requires=['numpy'],  # note: can't require torch here as it will install torch even for a TensorFlow container
    packages=setuptools.find_packages(),    
    python_requires='>=3.6',
)

# --- importing other libraries 

# Add line to .bashrc
subprocess.run('echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> ~/.bashrc', shell=True)

# Install libeigen3-dev
subprocess.run('sudo apt-get install -y libeigen3-dev', shell=True)

# Install libglu1-mesa-dev
subprocess.run('sudo apt-get install -y libglu1-mesa-dev', shell=True)

# Install libegl1-mesa-dev
subprocess.run('sudo apt-get install -y libegl1-mesa-dev', shell=True)

# Create symbolic link
subprocess.run('sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen', shell=True)