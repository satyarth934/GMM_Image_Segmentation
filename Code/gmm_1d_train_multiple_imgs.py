from __future__ import print_function, division
import os
import sys
import cv2
import copy
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
# np.seterr(divide='ignore', invalid='ignore')

import utils
import gmm1d


def main():
	"""
	Training for the orange_buoy
	Yellow buoy is trained using the red channel
	"""
	img_dir = "../Data/Proper_Dataset/orange_buoy/"

	mu, variance, weights = gmm1d.gmm1d(img_dir, num_gaussians=3, channel="red")

	gmm1d_orange_params = {}
	gmm1d_orange_params["mean"] = mu
	gmm1d_orange_params["variance"] = variance
	gmm1d_orange_params["weights"] = weights

	np.save("../Params/gmm1d_orange_params.npy", gmm1d_orange_params)

	gmm1d.testModel(frame_path="../Data/frame_set/buoy_frame_110.jpg", img_channel="red", model=(mu, variance, weights))

	"""
	Training for the green_buoy
	Yellow buoy is trained using the green channel
	"""
	img_dir = "../Data/Proper_Dataset/green_buoy/"

	mu, variance, weights = gmm1d.gmm1d(img_dir, num_gaussians=3, channel="green")

	gmm1d_green_params = {}
	gmm1d_green_params["mean"] = mu
	gmm1d_green_params["variance"] = variance
	gmm1d_green_params["weights"] = weights

	np.save("../Params/gmm1d_green_params.npy", gmm1d_green_params)

	gmm1d.testModel(frame_path="../Data/frame_set/buoy_frame_110.jpg", img_channel="green", model=(mu, variance, weights))

	"""
	Training for the yellow_buoy
	Yellow buoy is trained using the blue channel
	"""
	img_dir = "../Data/Proper_Dataset/yellow_buoy/"

	mu, variance, weights = gmm1d.gmm1d(img_dir, num_gaussians=3, channel="blue")

	gmm1d_yellow_params = {}
	gmm1d_yellow_params["mean"] = mu
	gmm1d_yellow_params["variance"] = variance
	gmm1d_yellow_params["weights"] = weights

	np.save("../Params/gmm1d_yellow_params.npy", gmm1d_yellow_params)

	gmm1d.testModel(frame_path="../Data/frame_set/buoy_frame_110.jpg", img_channel="blue", model=(mu, variance, weights))


if __name__ == '__main__':
	main()