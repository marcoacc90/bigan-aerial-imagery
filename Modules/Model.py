from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def make_generator_model(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_dim = latent_dim ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def generator_loss(fake_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output),logits=fake_output))

def make_discriminator_model( my_shape, latent_dim ):
    # D(x)
    img_input = keras.Input(shape=my_shape, name="img_input")
    x = layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=my_shape, name='conv1')( img_input )
    x = layers.LeakyReLU(name = 'leaky1')( x )
    x = layers.Dropout(0.3, name = 'dropout1')( x ) # training = is_training
    x = layers.Conv2D(128,(5,5),strides=(2,2),padding='same', name = 'conv2')( x )
    x = layers.LeakyReLU(name = 'leaky2')( x )
    x = layers.Dropout(0.3, name = 'dropout2')( x ) # training = is_training
    x  = layers.Flatten( name = 'flatten' )( x )
    # D(z)
    z_input = keras.Input( shape = latent_dim, name="z_input")
    z = layers.Dense(512, name = 'dense1')( z_input )
    z = layers.LeakyReLU( name = 'leaky3' )( z )
    z = layers.Dropout(0.3, name = 'dropout3')( z ) # training = is_training
    # Concatenate
    x = tf.concat([x, z], 1)
    x = layers.Dense(1024, name = 'dense2')( x )
    x = layers.LeakyReLU(name = 'leaky4')( x )
    x = layers.Dropout(0.3, name = 'dropout4')( x ) # training = is_training
    prediction = layers.Dense(1)( x )
    model = keras.Model(
        inputs=[img_input, z_input],
        outputs=[ prediction ],
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output),logits=real_output))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output),logits=fake_output))
    total_loss = real_loss + fake_loss
    return total_loss

def make_encoder_model( my_shape, latent_dim ):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=my_shape)) #1
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False )) #64
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)) #128
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Flatten())
	model.add(layers.Dense( latent_dim, activation='tanh'))
	opt = tf.keras.optimizers.RMSprop(1e-3)
	model.compile(optimizer= opt, loss='mean_squared_error')
	return model

#Tambien el encoder busca engañar
def encoder_loss( real_output ):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_output),logits=real_output))
