import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

# To avoid bad divisions
EPS = 0.000001


def computeGaussian(x, mu, sigma):
    return (1 / ((sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)) + EPS))


# Gaussian Function
def pdf(data, mean, std):
	variance = std**2
	# A normal continuous random variable.
	s1 = 1/(np.sqrt(2*np.pi*variance) + EPS)
	s2 = np.exp(-(np.square(data - mean)/(2*variance + EPS)))
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


def plotGaussians(hist, means, variances, plotter=plt):
	for i in range(len(means)):
		plotter.plot(hist, pdf(hist, means[i], np.sqrt(variances[i])), color="red")


def plotCombinedGaussians(hist, means, variances, weights, plotter=plt):
	likelihood = []
	for i in range(len(means)):
		likelihood.append(pdf(hist, means[i], np.sqrt(variances[i])))
	likelihood = np.array(likelihood)

	weights.shape = weights.shape[0], 1
	weighted_likelihood = weights * likelihood

	plotter.plot(hist, np.sum(weighted_likelihood, axis=0), color="green")


def generateInputData(img_dir, channel_idx):
	img_names = os.listdir(img_dir)
	img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]

	gen_data = np.array([])
	for img_path in img_paths:
		img = cv2.imread(img_path)
		img = img[:,:,channel_idx]
		img = img[img > IGNORE_THRESH].ravel()
		gen_data = np.append(gen_data, img)

	return gen_data


# def generateData(img_dir):
#     stack = []
#     for filename in os.listdir(img_dir):
#         image = cv2.imread(os.path.join(img_dir,filename))
#         resized = cv2.resize(image,(40,40),interpolation=cv2.INTER_LINEAR)
#         image = resized[13:27,13:27]
#         image = image[:,:,1]
#         ch = 1
#         nx = image.shape[0]
#         ny = image.shape[1]
#         image = np.reshape(image,(nx*ny,ch))
       
#         for i in range(image.shape[0]):
#             stack.append(image[i,:])
        
#     return np.array(stack)


def localizeBuoy(res_img):
	processed = cv2.medianBlur(res_img,3)
	processed = cv2.Canny(processed,20,255 )
	mask, cnts, h = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
	if len(cnts_sorted) > 0:
		hull = cv2.convexHull(cnts_sorted[0])
		# (x,y),radius = cv2.minEnclosingCircle(hull)
		return cv2.minEnclosingCircle(hull)
	else:
		return None