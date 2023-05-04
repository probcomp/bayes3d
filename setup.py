import setuptools
import subprocess
import os

setuptools.setup(
    name='bayes3d',
    version='1.0.0',
    package_data={  
        'bayes3d': [
            'nvdiffrast/common/*.h',
            'nvdiffrast/common/*.inl',
            'nvdiffrast/common/*.cu',
            'nvdiffrast/common/*.cpp',
            'nvdiffrast/lib/*.h',
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
print('Added line to .bashrc')

# Install libeigen3-dev
subprocess.run('sudo apt-get install -y libeigen3-dev', shell=True)
print('Installed libeigen3-dev')

# Install libglu1-mesa-dev
subprocess.run('sudo apt-get install -y libglu1-mesa-dev', shell=True)
print('Installed libglu1-mesa-dev')

# Install libegl1-mesa-dev
subprocess.run('sudo apt-get install -y libegl1-mesa-dev', shell=True)
print('Installed libegl1-mesa-dev')

# Create symbolic link
subprocess.run('sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen', shell=True)
print('Created symbolic link for Eigen')