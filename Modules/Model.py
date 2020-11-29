from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def make_encoder_model( latent_dim ):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',input_shape=[32, 32, 3]))

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))  #slope

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))  #slope

    model.add(layers.Flatten())
    model.add(layers.Dense( latent_dim ) ) #default linear? activation='tanh'

    opt = tf.keras.optimizers.RMSprop(1e-3)
    model.compile(optimizer= opt, loss='mean_squared_error')

    return model

def make_generator_model( latent_dim ):
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, use_bias=False, input_dim = latent_dim ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))  #slope

    model.add(layers.Dense(8*8*128, use_bias=False ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))  #slope

    model.add(layers.Reshape((8, 8, 128))) # Shape (None, 8, 8, 128)
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)) # Shape (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1)) #slope

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) # Shape 32,32,3
    model.add(layers.LeakyReLU(alpha=0.1)) #slope

    return model

#IMG_SHAPE = (32, 32, 3)
def make_discriminator_model( my_shape ):
    # D(x)
    img_input = keras.Input(shape=my_shape, name="img_input")
    z_input = keras.Input( shape = (200), name="z_input")

    x = layers.Conv2D(64,(4,4),strides=(2,2),padding='same',input_shape=my_shape, name='conv1')( img_input )
    x = layers.LeakyReLU( name = 'leaky1')( x )

    x = layers.Conv2D(64,(4,4),strides=(2,2),padding='same', name = 'conv2')( x )
    x = layers.BatchNormalization( name = 'batch1')( x )
    x = layers.LeakyReLU(name = 'leaky2')( x )
    # I think
    x  = layers.Flatten( name = 'flatten' )( x )

    # D(z)
    y = layers.Dense(512, name = 'dense1')( z_input )
    y = layers.LeakyReLU( name = 'leaky3' )( y )

    # Concatenate
    x = tf.concat([x, y], 1)

    x = layers.Dense(1024, name = 'dense2')( x )
    x = layers.LeakyReLU(name = 'leaky4')( x )
    
    prediction = layers.Dense(1, activation='sigmoid')( x )

    model = keras.Model(
        inputs=[img_input, z_input],
        outputs=[ prediction ],
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-5), # beta1 = 0.5
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def encoder_loss( real_output ):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_output),logits=real_output))

def generator_loss(fake_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output),logits=fake_output))

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output),logits=real_output))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output),logits=fake_output))

    total_loss = real_loss + fake_loss
    return total_loss
