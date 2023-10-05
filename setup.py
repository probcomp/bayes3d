#!/usr/bin/python

from setuptools import find_namespace_packages, setup


def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except Exception:
        return None


REQUIRED_PACKAGES = [
    'scikit-learn', 'wheel', 'ipython', 'jupyter', 'numpy', 'matplotlib',
    'pillow', 'opencv-python', 'open3d', 'graphviz', 'distinctipy', 'trimesh',
    'ninja', 'pyransac3d', 'meshcat', 'h5py', 'gdown', 'pytest', 'zmq',
    'plyfile', 'tqdm', 'jupyterlab', 'imageio', 'timm', 'joblib', 'pdoc3',
    'addict', 'tensorflow-probability', 'flax'
]

VERSION = '0.0.1'

setup(name='bayes3d',
      version=VERSION,
      description='Docker-based job runner for AI research.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      python_requires='>=3.10.0',
      author='Nishad Gothoskar',
      author_email='nishadg@mit.edu',
      url='https://github.com/probcomp/bayes3d',
      license='Apache-2.0',
      packages=find_namespace_packages(include=['bayes3d.*']),
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True)
