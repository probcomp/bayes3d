# From Gaussian splatting paper
# 
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
from convert_lib import convert_images_to_mesh


parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
# parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="SIMPLE_PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--single_camera", default=0, type=int)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
use_gpu = 1 if not args.no_gpu else 0

convert_images_to_mesh(args.source_path, colmap_command, args.camera, str(args.single_camera), str(use_gpu))
