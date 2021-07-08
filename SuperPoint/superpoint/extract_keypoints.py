from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from classes.utils import (AverageTimer, VideoStreamer,
						  make_matching_plot_fast, frame2tensor)


from classes.superpoint import SuperPoint

torch.set_grad_enabled(False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='SuperPoint extractor',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		'--input', type=str, default='0',
		help='ID of a USB webcam, URL of an IP camera, '
			 'or path to an image directory or movie file')
	parser.add_argument(
		'--output_dir', type=str, default=None,
		help='Directory where to write output frames (If None, no output)')

	parser.add_argument(
		'--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
		help='Glob if a directory of images is specified')
	parser.add_argument(
		'--skip', type=int, default=1,
		help='Images to skip if input is a movie or directory')
	parser.add_argument(
		'--max_length', type=int, default=1000000,
		help='Maximum length if input is a movie or directory')
	parser.add_argument(
		'--resize', type=int, nargs='+', default=[504, 521],
		help='Resize the input image before running inference. If two numbers, '
			 'resize to the exact dimensions, if one number, resize the max '
			 'dimension, if -1, do not resize')

	parser.add_argument(
 		'--clahe', type=int, default=1,
 		help='0 to use gray image, 1 to use CLAHE, 2 to use green channel')
	parser.add_argument(
 		'--gb', type=int, default=0,
 		help='Size of the Gaussian Filter')
	parser.add_argument(
 		'--clahe_size', type=int, default=3,
 		help='CLAHE tile grid size')
	parser.add_argument(
 		'--clahe_limit', type=int, default=3,
 		help='CLAHE clip limi')

	parser.add_argument(
		'--max_keypoints', type=int, default=-1,
		help='Maximum number of keypoints detected by Superpoint'
			 ' (\'-1\' keeps all keypoints)')
	parser.add_argument(
		'--keypoint_threshold', type=float, default=0.005,
		help='SuperPoint keypoint detector confidence threshold')
	parser.add_argument(
		'--nms_radius', type=int, default=4,
		help='SuperPoint Non Maximum Suppression (NMS) radius'
		' (Must be positive)')

	parser.add_argument(
		'--force_cpu', action='store_true',
		help='Force pytorch to run in CPU mode.')

	opt = parser.parse_args()
	print(opt)


	if len(opt.resize) == 2 and opt.resize[1] == -1:
		opt.resize = opt.resize[0:1]
	if len(opt.resize) == 2:
		print('Will resize to {}x{} (WxH)'.format(
			opt.resize[0], opt.resize[1]))
	elif len(opt.resize) == 1 and opt.resize[0] > 0:
		print('Will resize max dimension to {}'.format(opt.resize[0]))
	elif len(opt.resize) == 1:
		print('Will not resize images')
	else:
		raise ValueError('Cannot specify more than two integers for --resize')

	device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
	print('Running inference on device \"{}\"'.format(device))
	config = {
		'superpoint': {
			'nms_radius': opt.nms_radius,
			'keypoint_threshold': opt.keypoint_threshold,
			'max_keypoints': opt.max_keypoints
		}
	}

	superpoint = SuperPoint(config.get('superpoint', {}))
	keys = ['keypoints', 'scores', 'descriptors']

	# skip = 1
	# max_length = 1000000

	vs = VideoStreamer(opt.input, opt.resize, opt.skip,
					   opt.image_glob, opt.max_length)
	#
	# frame, ret = vs.next_frame()
	# assert ret, 'Error when reading the first frame (try different --input?)'
	#
	# frame_tensor = frame2tensor(frame, device)
	# last_data = superpoint({'image': frame_tensor})
	# last_data = {k+'0': last_data[k] for k in keys}
	# last_data['image0'] = frame_tensor
	# last_frame = frame
	# last_image_id = 0

	timer = AverageTimer()
	i = 0

	f_timestamps = open(opt.input + "/../times.txt")
	timestamps = f_timestamps.readlines()
	print(len(timestamps))
	while True:
		frame, ret = vs.next_frame()
		if not ret:
			print('Finished demo_superglue.py')
			break
		timer.update('data')
		# stem0, stem1 = last_image_id, vs.i - 1

		# cv2.imshow("image", frame)
		# cv2.waitKey(1)

		# CLAHE
		if opt.clahe == 1:

			# convert

			#cv2.imshow("original", frame)

			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#cv2.imshow("gray", gray)
			lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

			# split
			l,a,b = cv2.split(lab_image)

			# clahe
			cl = opt.clahe_limit
			c_size = opt.clahe_size
			clahe = cv2.createCLAHE(clipLimit =cl, tileGridSize=(c_size, c_size))
			l = clahe.apply(l)

			# merge
			lab_image = cv2.merge((l, a, b))

			# convert
			frame = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

			# green channel
			b, g, r = cv2.split(frame)
			frame = g
			if opt.gb != 0:
				frame = cv2.GaussianBlur(g,(opt.gb,opt.gb),0)
			

			#cv2.imshow("clahe", frame)
			#cv2.waitKey(1)
		elif opt.clahe == 2:
			# green channel
			b, g, r = cv2.split(frame)
			frame = g
		else:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame_tensor = frame2tensor(frame, device)
		pred = {}
		pred0 = superpoint({'image': frame_tensor})
		pred = {**pred, **{k: v for k, v in pred0.items()}}
		kpts = pred['keypoints'][0].cpu().numpy()
		desc = pred['descriptors'][0].cpu().numpy()
		timer.update('forward')


		# keypoints = [cv2.KeyPoint(p[0], p[1], 1) for p in kpts]
		# out = frame
		# out = cv2.drawKeypoints(frame, keypoints, out)
		# cv2.imshow("out", out)
		# cv2.waitKey(1)

		kpts = np.array([[p[0], p[1]] for p in kpts])
		#print(timestamps[i] + "\n")
		ts = timestamps[i][0:6]
		print(ts)
		# ts = "image"+str(i+1)
		ts = ts.rstrip("\n")
		np.savetxt(open(opt.output_dir + "/" + ts + "_kpts.txt", 'wb'), np.asarray(kpts))
		np.save(open(opt.output_dir + "/" + ts + "_kpts.npy", 'wb'), np.asarray(kpts))
		# print(ts)
		# print(desc.shape)
		# print(kpts.shape)
		desc = desc.transpose()
		#np.save(open(opt.output_dir + "/" + ts + "_desc.npy", 'wb'), desc)
		np.savetxt(open(opt.output_dir + "/" + ts + "_desc.txt", 'wb'), np.asarray(desc))

		if i % 100 == 0: print("Image " + str(i))

		i += 1

		timer.update('viz')
		timer.print()
