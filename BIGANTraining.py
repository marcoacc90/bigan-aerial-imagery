import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import os
import sys

def summarize_performance( path, epoch, g_model, e_model, d_model, images, latent_dim, n_samples=100 ) :
    xReal, yReal = TO.generateRealSamples( images, n_samples )
    E_x = e_model.predict( xReal )
    G_Ex = g_model.predict( E_x )
    z = tf.random.normal([n_samples, latent_dim], mean = 0.0, stddev = 1.0)
    G_z = g_model.predict( z )

    #Real images
    filename = '%s/plot_real_e%03d.png' % ( path, epoch+1 )
    X = TO.conver2image( xReal )
    TO.savePlot( filename, X )

    # Reconstructed images
    filename = '%s/plot_recon_e%03d.png' % ( path, epoch+1 )
    X = TO.conver2image( G_Ex )
    TO.savePlot( filename, X )

    # Generated images
    filename = '%s/plot_gen_e%03d.png' % ( path, epoch+1 )
    X = TO.conver2image( G_z )
    TO.savePlot( filename, X )

    # Save weights
    filename = '%s/encoder_weights_%03d' % (path, epoch+1)
    e_model.save_weights(filename)
    filename = '%s/generator_weights_%03d' % (path,epoch + 1)
    g_model.save_weights(filename)
    filename = '%s/discriminator_weights_%03d' % (path,epoch + 1)
    d_model.save_weights(filename)

@tf.function
def train_step( g_model, e_model, d_model, g_opt, e_opt, d_opt, x, latent_dim ):
    z = tf.random.normal([len(x), latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
        G_z = g_model( z, training=True )
        E_x = e_model( x, training=True )

        label_enc = d_model( {"img_input": x,  "z_input": E_x}, training = True)
        label_gen = d_model( {"img_input": G_z,"z_input": z  }, training = True)

        gen_loss = MO.generator_loss( label_gen )
        enc_loss = MO.encoder_loss( label_enc )
        disc_loss = MO.discriminator_loss( label_enc, label_gen )

    gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
    gradients_of_encoder = enc_tape.gradient(enc_loss, e_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

    g_opt.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
    e_opt.apply_gradients(zip(gradients_of_encoder, e_model.trainable_variables))
    d_opt.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))

def train( path, g_model, e_model, d_model, g_opt, e_opt, d_opt, dataset, train_images, epochs, latent_dim ):
  for epoch in range(epochs):

    start = time.time()
    for image_batch in dataset:
      train_step( g_model, e_model, d_model, g_opt, e_opt, d_opt, image_batch, latent_dim )
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    if (epoch + 1) % 100 == 0:
        summarize_performance( path, epoch, g_model, e_model, d_model, train_images, latent_dim )

# LOAD DATA FOR TRAINING
if len(sys.argv) != 3:
    print('python3 BIGANTraining.py <img_dir> <bigan_dir>\n')
    print('python3 BIGANTraining.py Dataset/normalTraining/ BIGAN_OUTPUT');
    sys.exit( 1 )

# LOAD THE PATCHES
PATCH_SIZE = GO.PATCH_SIZE

img_dir = str( sys.argv[ 1 ] )
bigan_dir = str( sys.argv[ 2 ] )

os.system('ls ' + img_dir + ' > Image.txt')
flag = False
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(img_dir + name)
        x = np.array( img )
        x = x.reshape( (1,PATCH_SIZE,PATCH_SIZE,3) )
        if flag == True:
            train_images = np.concatenate( (train_images,x), axis = 0)
        else:
            train_images = np.copy( x )
            flag = True
        print( train_images.shape )

os.system('rm -r Image.txt')
print('Patches are ready, shape: {}'.format(train_images.shape))
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize the images to [-1, 1]

BUFFER_SIZE = len(train_images)
BATCH_SIZE = GO.BATCH_SIZE
latent_dim = GO.NOISE_DIM
EPOCHS = GO.N_EPOCHS
image_dim = GO.IMAGE_DIM

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# NAME OF THE OUTPUT PATH
os.system( 'mkdir ' + bigan_dir )

print( 'Noise dim: ', noise_dim )
print( 'Image size:  ', image_dim )

# CREATE THE MODELS AND OPTIMIZERS
e_model = MO.make_encoder_model( image_dim, latent_dim )
g_model = MO.make_generator_model( image_dim, latent_dim )
d_model = MO.make_discriminator_model( image_dim, latent_dim )
e_opt = tf.keras.optimizers.Adam( 1e-4 )
g_opt = tf.keras.optimizers.Adam( 1e-4 )
d_opt = tf.keras.optimizers.Adam( 1e-4 )


# START THE TRAINING
train( bigan_dir, g_model, e_model, d_model, g_opt, e_opt, d_opt, dataset, train_images, EPOCHS, latent_dim )
