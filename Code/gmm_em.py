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


def getGMMs():
	np.random.seed(156)

	dir_path = sys.argv[1]
	imgs = os.listdir(dir_path)
	img_paths = [os.path.join(dir_path, _) for _ in imgs]
	img_paths.sort()

	k = 3
	weights = np.ones((k)) / k 		# equal weightage to all gaussians in the beginning
	means = np.random.choice(range(256), k) 	# k random means from the given dataset
	variances = np.random.random_sample(size=k) 	# k values from [0,1)

	for img_path in img_paths[:5]:
		print(img_path)
		img = cv2.imread(img_path)

		##
		## Extracting channels
		##
		blue = img[:, :, 0]
		green = img[:, :, 1]
		red = img[:, :, 2]

		# n_samples = 100
		# mu1, sigma1 = -4, 1.2 # mean and variance
		# mu2, sigma2 = 4, 1.8 # mean and variance
		# mu3, sigma3 = 0, 1.6 # mean and variance
		# blue = np.random.normal(mu1, np.sqrt(sigma1), n_samples)
		# green = np.random.normal(mu2, np.sqrt(sigma2), n_samples)
		# red = np.random.normal(mu3, np.sqrt(sigma3), n_samples)

		##
		## Flattening the single channel matrices
		##
		blue_flat = blue[blue>=IGNORE_THRESH].ravel()
		green_flat = green[green>=IGNORE_THRESH].ravel()
		red_flat = red[red>=IGNORE_THRESH].ravel()
		print("means:", [np.mean(blue_flat), np.mean(green_flat), np.mean(red_flat)])
		print("variances:", [np.std(blue_flat)**2, np.std(green_flat)**2, np.std(red_flat)**2])
		print("------------")

		##
		## Computing histograms for each channels
		##
		fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
		# blue_n, blue_hist, _ = ax[0].hist(blue.ravel(), 256 - IGNORE_THRESH, range=[IGNORE_THRESH, 256], normed=True, color="blue")
		blue_n, blue_hist, _ = ax[0].hist(blue_flat, 256, range=[0, 256], normed=True, color="blue")
		green_n, green_hist, _ = ax[1].hist(green_flat, 256, range=[0, 256], normed=True, color="green")
		red_n, red_hist, _ = ax[2].hist(red_flat, 256, range=[0, 256], normed=True, color="red")
		
		blue_data = (blue_hist, np.mean(blue_flat), np.std(blue_flat))
		green_data = (green_hist, np.mean(green_flat), np.std(green_flat))
		red_data = (red_hist, np.mean(red_flat), np.std(red_flat))
		utils.visGaussianCurve(blue_data, green_data, red_data, plotter=ax[3])
		
		##
		## Training data visualization
		##
		# plt.figure("training data vis")
		# utils.dataVisualization(blue_flat, blue_hist, green_flat, green_hist, red_flat, red_hist)
		plt.show()

		##
		## define the number of clusters to be learned
		##
		img_flat = np.array(list(blue_flat) + list(green_flat) + list(red_flat))
		np.random.shuffle(img_flat)
		#######################################################################
		## k = 3
		## weights = np.ones((k)) / k 		# equal weightage to all gaussians in the beginning
		## means = np.random.choice(img_flat, k) 	# k random means from the given dataset
		## variances = np.random.random_sample(size=k) 	# k values from [0,1)
		## print(means, variances)
		## print(img_flat.shape)
		#######################################################################


		plt.figure(figsize=(10,6))
		plt.ion()
		##
		## EM --- Expectation Maximization Step
		##
		eps=1e-8 	# just to avoid division by 0 error
		for step in range(100):
			if step % 1 == 0:
				axes = plt.gca()

				plt.xlabel("$x$")
				plt.ylabel("pdf")
				plt.title("Iteration {}".format(step))
				# ground truth curves
				blue_data = (blue_hist, np.mean(blue_flat), np.std(blue_flat))
				green_data = (green_hist, np.mean(green_flat), np.std(green_flat))
				red_data = (red_hist, np.mean(red_flat), np.std(red_flat))
				utils.visGaussianCurve(blue_data, green_data, red_data, plotter=axes, colors=['grey']*3, labels=['True pdf']*3)
				# plt.plot(bins, pdf(bins, mu1, sigma1), color='grey', label="True pdf")
				# plt.plot(bins, pdf(bins, mu2, sigma2), color='grey')
				# plt.plot(bins, pdf(bins, mu3, sigma3), color='grey')

				# data points
				plt.scatter(img_flat, [0.005] * len(img_flat), color='navy', s=30, marker=2, label="Train data")

				# modeled curves 
				blue_data = (blue_hist, means[0], np.sqrt(variances[0]))
				green_data = (green_hist, means[1], np.sqrt(variances[1]))
				red_data = (red_hist, means[2], np.sqrt(variances[2]))
				utils.visGaussianCurve(blue_data=blue_data, green_data=green_data, red_data=red_data, plotter=axes, colors=['blue', 'green', 'red'], labels=['cluster 1', 'cluster 2', 'cluster 3'])
				# plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue', label="Cluster 1")
				# plt.plot(bins, pdf(bins, means[1], variances[1]), color='green', label="Cluster 2")
				# plt.plot(bins, pdf(bins, means[2], variances[2]), color='magenta', label="Cluster 3")
				
				# plt.legend(loc='upper left')
				
				# plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
				plt.show()
				plt.pause(0.0001)

			else:
				print("step % 1 is not = 0 for some weird reason!!!")
			
			# calculate the maximum likelihood of each observation xi
			likelihood = []
			# Expectation step
			for j in range(k):
				likelihood.append(utils.pdf(img_flat, means[j], np.sqrt(variances[j])))
			likelihood = np.array(likelihood)
				
			b = []
			# Maximization step 
			for j in range(k):
				# use the current values for the parameters to evaluate the posterior
				# probabilities of the data to have been generanted by each gaussian    
				b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))
			
				# updage mean and variance
				means[j] = np.sum(b[j] * img_flat) / (np.sum(b[j]+eps))
				variances[j] = np.sum(b[j] * np.square(img_flat - means[j])) / (np.sum(b[j]+eps))
				
				# update the weights
				weights[j] = np.mean(b[j])

			# plt.clf()
		plt.ioff()
		plt.show()

	return (weights, means, variances)


def main():
	weights, means, variances = getGMMs()
	print("weights:", weights)
	print("means:", means)
	print("variances:", variances)
	print("===============")


if __name__ == "__main__":
	main()


# image as input
# compute mean on all channels
# compute std on all channels
# compute probability for all pixels for each gaussian. this is bayes rule.
# compute gaussian using all the pixels. (but it is weight)
# after it is done for all pixels, the gaussian mean and std is updated
#
# mean is updated to be weighted average and the std is updated to be the weighted std
