import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True

import utils


def function():
    data = np.random.normal(size=500)

    plt.scatter(x=data, y=[0.0] * len(data))
    # plt.scatter(data, , s=30, marker=2)
    plt.plot(data, utils.pdf(data, mean=np.mean(data), variance=np.std(data)))

    plt.show()


def main():
    function()


if __name__ == '__main__':
    main()
