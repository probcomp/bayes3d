# Kubric Rendering Pipeline

## Setup

### Kubric Dependencies

First clone the Kubric git repository 

```git clone https://github.com/google-research/kubric.git```

Then use 'pip' to install the non-'bpy' dependencies

```pip install -r kubric/requirements.txt```

### Blender Install

Next install the Blender python module 'bpy' by [building Blender as a python module](https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule). Use the local install options to install the module. 

### OpenEXR Install 

Get the installers for 'libopenexr' and 'openexr'

```sudo apt-get install libopenexr-dev```

```sudo apt-get install openexr```

Then install openexr python module with pip 

```pip install OpenEXR --user```

### Kubric Install 

Finally install Kubric with pip from the source directory

```pip install .```

Kubric worker files can then be run normally using python

```python3 examples/helloworld.py```

## Example 

From the source directory run the sample scene

```python3 jax3dp3/kubric_render/sample_scene.py```

