import matplotlib.pyplot as plt
import numpy as np
from time import time
import os

def show_image(image):
    plt.imshow(image.reshape(28, 28), cmap = 'binary')
    plt.show()

def plot_image_label_prediction(images, labels, prediction = [], idx = 0, num = 20):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(np.reshape(images[idx], (28, 28)), cmap="binary")
        title = "label = " + str(np.argmax(labels[idx]))
        if len(prediction) > 0:
            title += ", prediction = " + str(prediction[idx])
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
