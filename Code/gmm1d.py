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


def gmm1dUtils(data, num_gaussians, animate=False):
	if animate:
		n, hist, _ = plt.hist(data, utils.BINS, range=[utils.RANGE_MIN, utils.RANGE_MAX], normed=True, color="blue"
			)
	
	weights = np.ones((num_gaussians)) / num_gaussians 		# init equal weightage to all gaussians
	means = np.random.choice(data, num_gaussians) 	# k random means from the given dataset
	meansPrev = np.zeros(num_gaussians) 	# k random means from the given dataset
	variances = np.random.random_sample(size=num_gaussians) 	# k values from [0,1)

	print("--- Initializations ---")
	print("weights:", weights)
	print("means:", means)
	print("variances:", variances)
	print("=======================")

	if animate:
		plt.ion()

	log_likelihood_thresh = 0.000001
	log_likelihood_prev = 0

	max_itr = 100
	itr = 0
	while itr < max_itr:

		# print("data:", data.shape)

		# Expectation
		likelihood = []
		for i in range(num_gaussians):
			likelihood.append(utils.pdf(data, means[i], np.sqrt(variances[i])))
		likelihood = np.array(likelihood)
		# print("likelihood:", likelihood.shape)

		weights.shape = num_gaussians, 1
		weighted_likelihood = weights * likelihood
		# print("weighted_likelihood:", weighted_likelihood.shape)
		# print("np.sum(weighted_likelihood, axis=0):", np.sum(weighted_likelihood, axis=0).shape)

		cluster_prob = weighted_likelihood / (np.sum(weighted_likelihood, axis=0) + utils.EPS) 	# P(b|xi)
		# print("cluster_prob:", cluster_prob.shape)

		meansPrev = means
		means = np.sum(cluster_prob * data, axis=1) / (np.sum(cluster_prob, axis=1) + utils.EPS)
		# print("means:", means.shape)
		# print(means)

		means.shape = len(means), 1
		# print("means:", means.shape)

		sqdiff = np.subtract(data, means)**2
		variances = np.sum(cluster_prob * sqdiff, axis=1) / (np.sum(cluster_prob, axis=1) + utils.EPS)
		# print(variances.shape)

		weights = np.sum(cluster_prob, axis=1) / len(data)
		
		if animate:
			plt.title("Iteration {}".format(itr))
			utils.plotGaussians(hist, means, variances)
			plt.show()
			plt.pause(0.001)

		# print("cluster_prob.shape:", cluster_prob.shape)
		# cluster_sum = np.sum(cluster_prob,axis=0)
		# print("cluster_sum.shape:", cluster_sum.shape)
		# log_likelihood = np.sum(np.log(cluster_sum))
		# print("log_likelihood:", log_likelihood)
		if np.sum(np.abs(meansPrev - means)) < 0.1:
			break

		itr += 1
	if animate:
		plt.ioff()
		plt.show()

	return means, variances, weights

	
def gmm1d(img_dir, num_gaussians, channel):	
	if channel in ["blue", "Blue", "BLUE", "b", "B"]:
		channel_idx = 0
	elif channel in ["green", "Green", "GREEN", "g", "G"]:
		channel_idx = 1
	else:
		channel_idx = 2

	input_data = utils.generateInputData(img_dir, channel_idx)
	# input_data = utils.generateData(img_dir)
	print("input_data.shape:", input_data.shape)
	# sys.exit(0)

	# input_data = img[:,:,channel_idx]
	# input_data = input_data[input_data>utils.IGNORE_THRESH].ravel()

	plt.figure("convergence")
	mu, variance, weights = gmm1dUtils(input_data, num_gaussians=3, animate=True)	# Using only Blue Channel

	print("---------- RESULTS -----------")
	print("mu:", mu)
	print("variance:", variance)
	print("weights:", weights)

	n, hist, _ = plt.hist(input_data, utils.BINS, range=[utils.RANGE_MIN, utils.RANGE_MAX], normed=True, color="blue")

	utils.plotGaussians(hist, mu, variance)
	utils.plotCombinedGaussians(hist, mu, variance, weights)
	plt.show()

	return mu, variance, weights


def gmm1dInference(test_img, model, channel="red"):
	mu = model[0]
	variance = model[1]
	weights = model[2]

	if channel in ["blue", "Blue", "BLUE", "b", "B"]:
		channel_idx = 0
	elif channel in ["green", "Green", "GREEN", "g", "G"]:
		channel_idx = 1
	else:
		channel_idx = 2

	input_data = test_img[:,:,channel_idx]
	input_data = input_data.ravel()
	print(input_data.shape)
	num_gaussians = len(mu)

	likelihood = []
	for i in range(num_gaussians):
		likelihood.append(utils.pdf(input_data, mu[i], np.sqrt(variance[i])))
	likelihood = np.array(likelihood)

	weights.shape = num_gaussians, 1
	weighted_likelihood = weights * likelihood

	probabilities = np.sum(weighted_likelihood, axis=0)
	probabilities = np.reshape(probabilities, (test_img[:,:,channel_idx].shape))
	print("probabilities:", probabilities)

	# probabilities[probabilities>np.max(probabilities)/2.0] = 255
	probabilities[probabilities>(np.median(probabilities)*2)] = 255
	probabilities[probabilities!=255] = 0
	# plt.figure("mask"); plt.imshow(probabilities)

	output = np.zeros((test_img.shape[0], test_img.shape[1], 3))
	output[:,:,channel_idx] = probabilities
	# cv2.imshow(winname="output", mat=output)
	# cv2.waitKey(0)

	# plt.figure("output"); plt.imshow(output)
	plt.show()

	return (probabilities.astype(np.uint8), output)


def testModel(frame_path, img_channel, model):
	# frame_path = "../Data/frame_set/buoy_frame_0.jpg"
	# img_channel = "red" | "green" | "blue"
	mu = model[0]
	variance = model[1]
	weights = model[2]

	test_img = cv2.imread(frame_path)
	res_mask, res_img = gmm1dInference(test_img, model=(mu, variance, weights), channel=img_channel)
	localized_buoy = utils.localizeBuoy(res_img.astype(np.uint8))	
	if localized_buoy is not None:
		(x,y),radius = localized_buoy
		print(x,y,radius)

		if radius > utils.MIN_RADIUS_THRESH:
			cv2.circle(test_img, (int(x),int(y)-1),int(radius+1), utils.CHANNEL_COLORS["red"], 3)
			cv2.imshow("Final output",test_img)
			# images.append(test_img)
		else:
			cv2.imshow("Final output",test_img)
			# images.append(test_img)
		cv2.waitKey(0)
	else:
		"No detection!!"