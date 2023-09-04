DATASET_PATH=ku_scene_vids_linear_57/frames1
#colmap automatic_reconstructor --workspace_path $DATASET_PATH --image_path $DATASET_PATH/images
colmap model_converter --input_path $DATASET_PATH/dense/0/sparse --output_path $DATASET_PATH/dense/0/sparse --output_type TXT
colmap delaunay_mesher --input_path $DATASET_PATH/dense/0 --output_path $DATASET_PATH/dense/meshed_delaunay.ply
