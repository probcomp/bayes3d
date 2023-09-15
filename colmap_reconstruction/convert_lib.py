import os
import logging
import shutil


def convert_images_to_mesh(source_path, colmap_command = '', camera = 'SIMPLE_PINHOLE', single_camera = '0', sequential_frames = '0', use_gpu = '1'):
    
    #os.makedirs(source_path + "/colmap/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/database.db \
        --image_path " + source_path + "/images \
        --ImageReader.single_camera " + single_camera + " \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.use_gpu " + use_gpu # single camera set back to 1 for synthetic data
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching

    # nonsequential matching
    if not sequential_frames:
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + source_path + "/database.db \
            --SiftMatching.use_gpu " + use_gpu
        exit_code = os.system(feat_matching_cmd)
    
    # sequential matching (data output from video)
    else:
        feat_matching_cmd = colmap_command + " sequential_matcher \
            --database_path " + source_path + "/database.db \
            --SiftMatching.use_gpu " + use_gpu
        exit_code = os.system(feat_matching_cmd)

    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)


    if not os.path.exists(source_path + "/sparse"):
        os.mkdir(source_path + "/sparse")

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/database.db \
        --image_path "  + source_path + "/images \
        --output_path "  + source_path + "/sparse")
        #--Mapper.ba_global_function_tolerance=0.000001") # leave as default value
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    if not os.path.exists(source_path + "/dense"):
        os.mkdir(source_path + "/dense")

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + source_path + "/images \
        --input_path " + source_path + "/sparse/0 \
        --output_path " + source_path + "/dense \
        --output_type COLMAP \
        --max_image_size 2000")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)


    ### Patch match stereo
    img_patchmatchstero_cmd = (colmap_command + " patch_match_stereo \
        --workspace_path " + source_path + "/dense \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true")
    exit_code = os.system(img_patchmatchstero_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Stereo fusion
    img_stereofusion_cmd = (colmap_command + " stereo_fusion \
        --workspace_path " + source_path + "/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path " + source_path + "/dense/fused.ply")
    exit_code = os.system(img_stereofusion_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Convert binary COLMAP models to TXT

    convert_cmd = (colmap_command + " model_converter \
        --input_path " + source_path + "/dense/sparse \
        --output_path " + source_path + "/dense/sparse \
        --output_type TXT")
    exit_code = os.system(convert_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Generate Poisson mesh
    img_poissonmesher_cmd = (colmap_command + " poisson_mesher \
        --input_path " + source_path + "/dense/fused.ply \
        --output_path " + source_path + "/dense/meshed-poisson.ply")
    exit_code = os.system(img_poissonmesher_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Generate Delaunay mesh
    img_delaunaymesher_cmd = (colmap_command + " delaunay_mesher \
        --input_path " + source_path + "/dense \
        --output_path " + source_path + "/dense/meshed-delaunay.ply")
    exit_code = os.system(img_delaunaymesher_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print("Done.")
