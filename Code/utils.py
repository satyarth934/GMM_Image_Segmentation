import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def computeGaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)))


# Gaussian Function
def pdf(data, mean, std):
	variance = std**2
	# A normal continuous random variable.
	s1 = 1/(np.sqrt(2*np.pi*variance))
	s2 = np.exp(-(np.square(data - mean)/(2*variance)))
	return s1 * s2


# def visGaussianCurve(blue_flat, blue_hist, green_flat, green_hist, red_flat, red_hist, plotter, colors, labels):
def visGaussianCurve(blue_data, green_data, red_data, plotter, colors=None, labels=None):

	if blue_data is not None:
		blue_hist = blue_data[0]
		blue_mean = blue_data[1]
		blue_std = blue_data[2]
		plotter.plot(blue_hist, pdf(blue_hist, blue_mean, blue_std), color=('blue' if colors is None else colors[0]), label=("True pdf - blue" if labels is None else labels[0]))

	if green_data is not None:
		green_hist = green_data[0]
		green_mean = green_data[1]
		green_std = green_data[2]
		plotter.plot(green_hist, pdf(green_hist, green_mean, green_std), color=('green' if colors is None else colors[1]), label=("True pdf - green" if labels is None else labels[1]))

	if red_data is not None:
		red_hist = red_data[0]
		red_mean = red_data[1]
		red_std = red_data[2]
		plotter.plot(red_hist, pdf(red_hist, red_mean, red_std), color=('red' if colors is None else colors[2]), label=("True pdf - red" if labels is None else labels[2]))


def dataVisualization(blue_flat, blue_hist, green_flat, green_hist, red_flat, red_hist):
	"""
	Training data visualization
	"""
	plt.xlabel("$x$")
	plt.ylabel("pdf")
	
	plt.scatter(blue_flat, [0.001] * len(blue_flat), color='blue', s=30, marker=2, label="Train data - blue")
	plt.scatter(green_flat, [0.002] * len(green_flat), color='green', s=30, marker=2, label="Train data - green")
	plt.scatter(red_flat, [0.003] * len(red_flat), color='red', s=30, marker=2, label="Train data - red")

	img_flat = list(blue_flat) + list(green_flat) + list(red_flat)
	np.random.shuffle(img_flat)
	plt.scatter(img_flat, [0.005] * len(img_flat), color='black', s=30, marker=2, label="Train data - all")

	plt.plot(blue_hist, pdf(blue_hist, np.mean(blue_flat), np.std(blue_flat)), color='blue', label="True pdf - blue")
	plt.plot(green_hist, pdf(green_hist, np.mean(green_flat), np.std(green_flat)), color='green', label="True pdf - green")
	plt.plot(red_hist, pdf(red_hist, np.mean(red_flat), np.std(red_flat)), color='red', label="True pdf - red")

	plt.legend()
	plt.plot()


