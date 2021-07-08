#!/bin/bash

. ~/activate_conda.sh r2d2_env
cd  ORB-SLAM3-MIS/SuperPoint/superpoint


# CLAHE (3, 3x3) 
python extract_keypoints.py  --input ../../../Datasets/syn_kidney/manually_texturized/new/ --keypoint_threshold 0.0002 --output_dir ../../../Datasets/syn_kidney/manually_texturized/new/superpoint_3_3_no/ --clahe 1 --clahe_size 3 --clahe_limit 3 --max_keypoints 1000

# CLAHE (3, 3x3) + GB 5x5
python extract_keypoints.py  --input ../../../Datasets/syn_kidney/manually_texturized/new/ --keypoint_threshold 0.0002 --output_dir ../../../Datasets/syn_kidney/manually_texturized/new/superpoint_3_3_5/ --clahe 1 --clahe_size 3 --clahe_limit 3 --gb 5 --max_keypoints 1000

# CLAHE (3, 3x3) + GB 7x7
python extract_keypoints.py  --input ../../../Datasets/syn_kidney/manually_texturized/new/ --keypoint_threshold 0.0002 --output_dir ../../../Datasets/syn_kidney/manually_texturized/new/superpoint_3_3_7/ --clahe 1 --clahe_size 3 --clahe_limit 3 --gb 7 --max_keypoints 1000

# CLAHE (3, 8x8) 
python extract_keypoints.py  --input ../../../Datasets/syn_kidney/manually_texturized/new/ --keypoint_threshold 0.0002 --output_dir ../../../Datasets/syn_kidney/manually_texturized/new/superpoint_3_8_no/ --clahe 1 --clahe_size 8 --clahe_limit 3 --max_keypoints 1000

# CLAHE (3, 8x8) + GB 5x5 
python extract_keypoints.py  --input ../../../Datasets/syn_kidney/manually_texturized/new/ --keypoint_threshold 0.0002 --output_dir ../../../Datasets/syn_kidney/manually_texturized/new/superpoint_3_8_5/ --clahe 1 --clahe_size 8 --clahe_limit 3 --gb 5 --max_keypoints 1000

# CLAHE (3, 8x8) + GB 7x7
python extract_keypoints.py  --input ../../../Datasets/syn_kidney/manually_texturized/new/ --keypoint_threshold 0.0002 --output_dir ../../../Datasets/syn_kidney/manually_texturized/new/superpoint_3_8_7/ --clahe 1 --clahe_size 8 --clahe_limit 3 --gb 7 --max_keypoints 1000