from __future__ import print_function
import os
import sys
import cv2
import copy
import numpy as np
from imutils import contours
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True

import utils


# Pixels with intensities below this value are ignored
IGNORE_THRESH = 50

# Bins and Range
BINS = 256
RANGE_MIN = 0
RANGE_MAX = 256

# Channel Colors
CHANNEL_COLORS = {"blue": (255,0,0), "green": (0,255,0), "red": (0,0,255)}

# Minimum buoy radius when localizing contours
MIN_RADIUS_THRESH = 5

def gmm1dUtils(data, num_gaussians, animate=False):
	if animate:
		n, hist, _ = plt.hist(data, BINS, range=[RANGE_MIN, RANGE_MAX], normed=True, color="blue"
			)
	
	weights = np.ones((num_gaussians)) / num_gaussians 		# init equal weightage to all gaussians
	means = np.random.choice(data, num_gaussians) 	# k random means from the given dataset
	meansPrev = np.zeros(num_gaussians) 	# k random means from the given dataset
	variances = np.random.random_sample(size=num_gaussians) 	# k values from [0,1)

	eps = 0.000001 	# To avoid bad divisions

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

		cluster_prob = weighted_likelihood / np.sum(weighted_likelihood, axis=0) + eps 	# P(b|xi)
		# print("cluster_prob:", cluster_prob.shape)

		meansPrev = means
		means = np.sum(cluster_prob * data, axis=1) / np.sum(cluster_prob, axis=1) + eps
		# print("means:", means.shape)
		# print(means)

		means.shape = len(means), 1
		# print("means:", means.shape)

		sqdiff = np.subtract(data, means)**2
		variances = np.sum(cluster_prob * sqdiff, axis=1) / np.sum(cluster_prob, axis=1) + eps
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

	
def gmm1d(img, num_gaussians, channel):	
	if channel in ["blue", "Blue", "BLUE", "b", "B"]:
		channel_idx = 0
	elif channel in ["green", "Green", "GREEN", "g", "G"]:
		channel_idx = 1
	else:
		channel_idx = 2

	input_data = img[:,:,channel_idx]
	input_data = input_data[input_data>IGNORE_THRESH].ravel()

	plt.figure("convergence")
	mu, variance, weights = gmm1dUtils(input_data, num_gaussians=3, animate=False)	# Using only Blue Channel

	print("---------- RESULTS -----------")
	print("mu:", mu)
	print("variance:", variance)
	print("weights:", weights)

	n, hist, _ = plt.hist(input_data, BINS, range=[RANGE_MIN, RANGE_MAX], normed=True, color="blue")

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

	probabilities[probabilities>np.max(probabilities)/2.0] = 255
	probabilities[probabilities!=255] = 0
	# plt.figure("mask"); plt.imshow(probabilities)

	output = np.zeros((test_img.shape[0], test_img.shape[1], 3))
	output[:,:,channel_idx] = probabilities
	# cv2.imshow(winname="output", mat=output)
	# cv2.waitKey(0)

	# plt.figure("output"); plt.imshow(output)
	plt.show()

	return (probabilities.astype(np.uint8), output)


def main():
	img = cv2.imread("../Data/Proper_Dataset/orange_buoy/orange_1.jpg")



	mu, variance, weights = gmm1d(img, num_gaussians=3, channel="red")

	# test_img = cv2.imread("../Data/Proper_Dataset/orange_buoy/orange_15.jpg")
	test_img = cv2.imread("../Data/frame_set/buoy_frame_0.jpg")
	res_mask, res_img = gmm1dInference(test_img, model=(mu, variance, weights), channel="red")

	final = test_img[:,:,2] * (res_mask / 255)

	processed = cv2.medianBlur(res_img.astype(np.uint8),3)
	processed = cv2.Canny(processed,20,255 )
	mask, cnts, h = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
	hull = cv2.convexHull(cnts_sorted[0])
	(x,y),radius = cv2.minEnclosingCircle(hull)
	
	print(x,y,radius)

	if radius > MIN_RADIUS_THRESH:
		cv2.circle(test_img, (int(x),int(y)-1),int(radius+1), CHANNEL_COLORS["red"], 3)
		cv2.imshow("Final output",test_img)
		# images.append(test_img)
	else:
		cv2.imshow("Final output",test_img)
		# images.append(test_img)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()