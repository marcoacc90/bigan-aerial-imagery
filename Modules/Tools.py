from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
import numpy as np

def conver2image( images ) :
	X = (images + 1.0) / 2.0
	return X


def generateRealSamples( images, n_samples):
	ix = np.random.randint(0, images.shape[0], n_samples )
	X = images[ ix ]
	y = np.ones((n_samples, 1))
	return X, y

def generateFakeSamplesUniform( g_model, n_samples, latent_dim, min, max):
    x_input = tf.random.uniform([n_samples, latent_dim],minval = min, maxval = max)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def generateFakeSamplesNormal( g_model, n_samples, latent_dim, m = 0.0, s = 1.0):
    x_input = tf.random.normal([n_samples, latent_dim], mean = m, stddev = s)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def savePlot( name, examples, n=10 ):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i )
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, :])   #, cmap='rgb'
    filename = '%s' % ( name )
    pyplot.savefig(filename)
    pyplot.close()
