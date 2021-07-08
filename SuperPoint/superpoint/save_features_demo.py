import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf  # noqa: E402

from superpoint.settings import EXPER_PATH  # noqa: E402

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
												 keep_k_points=1000):

	def select_k_best(points, k):
		""" Select the k most probable points (and strip their proba).
		points has shape (num_points, 3) where the last coordinate is the proba. """
		sorted_prob = points[points[:, 2].argsort(), :2]
		# print(points.shape)
		start = min(k, points.shape[0])
		return sorted_prob[-start:, :]

	# Extract keypoints
	keypoints = np.where(keypoint_map > 0)
	prob = keypoint_map[keypoints[0], keypoints[1]]
	keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

	keypoints = select_k_best(keypoints, keep_k_points)
	keypoints = keypoints.astype(int)

	# Get descriptors for keypoints
	desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

	# Convert from just pts to cv2.KeyPoints
	keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

	return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
	# Match the keypoints with the warped_keypoints with nearest neighbor search
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(desc1, desc2)
	matches_idx = np.array([m.queryIdx for m in matches])
	m_kp1 = [kp1[idx] for idx in matches_idx]
	matches_idx = np.array([m.trainIdx for m in matches])
	m_kp2 = [kp2[idx] for idx in matches_idx]

	return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
	matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
	matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

	# Estimate the homography between the matches using RANSAC
	H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
									matched_pts2[:, [1, 0]],
									cv2.RANSAC)
	inliers = inliers.flatten()
	return H, inliers


def preprocess_image(img_file, img_size):
	img = cv2.imread(img_file, cv2.IMREAD_COLOR)
	img = cv2.resize(img, img_size)
	img_orig = img.copy()

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.expand_dims(img, 2)
	img = img.astype(np.float32)
	img_preprocessed = img / 255.

	return img_preprocessed, img_orig


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser(description='Compute the homography \
			between two images with the SuperPoint feature matches.')
	parser.add_argument('weights_name', type=str)
	parser.add_argument('--H', type=int, default=480,
						help='The height in pixels to resize the images to. \
								(default: 480)')
	parser.add_argument('--W', type=int, default=640,
						help='The width in pixels to resize the images to. \
								(default: 640)')
	parser.add_argument('--k_best', type=int, default=1000,
						help='Maximum number of keypoints to keep \
						(default: 1000)')
	parser.add_argument('--folder', type=str, default='.',
						help='Path to the folder that contains the images')
	parser.add_argument('--output', type=str, default='.',
						help='Path to the output folder')
	parser.add_argument("--format", type=str, default="png",
						help='image format')
	args = parser.parse_args()

	weights_name = args.weights_name
	img_size = (args.W, args.H)
	keep_k_best = args.k_best
	folder = args.folder
	output_path = args.output

	weights_root_dir = Path(EXPER_PATH, 'saved_models')
	weights_root_dir.mkdir(parents=True, exist_ok=True)
	weights_dir = Path(weights_root_dir, weights_name)

	graph = tf.Graph()
	with tf.Session(graph=graph) as sess:
		tf.saved_model.loader.load(sess,
								   [tf.saved_model.tag_constants.SERVING],
								   str(weights_dir))

		input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
		output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
		output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

		# read filenames
		f_timestamps = open(args.folder + "/timestamps.txt")
		timestamps = f_timestamps.readlines()
		for timestamp in timestamps:
		# for i in range(1,16384):
		# 	if i % 100 == 0: print("Image " + str(i))
		# 	timestamp = "image" + str(i)

			timestamp = timestamp.rstrip("\n")
			img_path = args.folder + "/" + timestamp + "." + args.format

			img, img_orig = preprocess_image(img_path, img_size)
			out = sess.run([output_prob_nms_tensor, output_desc_tensors],
							feed_dict={input_img_tensor: np.expand_dims(img, 0)})
			keypoint_map = np.squeeze(out[0])
			descriptor_map = np.squeeze(out[1])
			kpts, desc = extract_superpoint_keypoints_and_descriptors(
					keypoint_map, descriptor_map, keep_k_best)

			# print(kpts)

			kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
					  kp.angle, kp.response, kp.octave,
					  kp.class_id]
					 for kp in kpts])
			desc = np.array(desc)
			# print(kpts.size)
			# print(kpts)

			np.savetxt(open(args.output + "/" + timestamp + "_kpts.txt", 'wb'), np.asarray(kpts))
			np.save(open(args.output + "/" + timestamp + "_desc.npy", 'wb'), np.asarray(desc))
