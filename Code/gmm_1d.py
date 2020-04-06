from __future__ import print_function
import os
import sys
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True

import utils


# Pixels with intensities below this value are ignored
IGNORE_THRESH = 50

# Bins and Range
BINS = 256
RANGE_MIN = 0
RANGE_MAX = 256


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
	mu, variance, weights = gmm1dUtils(input_data, num_gaussians=3, animate=True)	# Using only Blue Channel

	print("---------- RESULTS -----------")
	print("mu:", mu)
	print("variance:", variance)
	print("weights:", weights)

	n, hist, _ = plt.hist(input_data, BINS, range=[RANGE_MIN, RANGE_MAX], normed=True, color="blue")

	utils.plotGaussians(hist, mu, variance)
	plt.show()


def main():
	img = cv2.imread("../Data/Proper_Dataset/orange_buoy/orange_1.jpg")
	gmm1d(img, num_gaussians=3, channel="red")


if __name__ == '__main__':
	main()