import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import os
import sys

if len(sys.argv) != 1:
    print('python3 BIGANTest.py\n')
    print('python3 BIGANTest.py');
    sys.exit( 1 )

# LOAD MODELS
EPOCHS = GO.N_EPOCHS
mode = 'Test'  # Validation,Test
latent_dim = GO.NOISE_DIM
folder = 'Dataset32x32/'
MODEL = 'BIGAN_' + str(EPOCHS)
alpha = 0.9

path = 'E%d_Results' % (EPOCHS)
cmd = 'mkdir ' + path
os.system( cmd )

name = 'E' + str(EPOCHS) +'_BIGAN/encoder_weights_' + '%03d' % (EPOCHS)
e_model = MO.make_encoder_model( GO.NOISE_DIM )
e_model.load_weights( name )

name = 'E' + str(EPOCHS) +'_BIGAN/generator_weights_' + '%03d' % (EPOCHS)
g_model = MO.make_generator_model( GO.NOISE_DIM )
g_model.load_weights( name )

name = 'E' + str(EPOCHS) +'_BIGAN/discriminator_weights_' + '%03d' % (EPOCHS)
d_model = MO.make_discriminator_model( GO.IMAGE_DIM )
d_model.load_weights( name )

d_features = tf.keras.Model( d_model.inputs, d_model.get_layer('leaky4').output )

PATCH_SIZE = GO.PATCH_SIZE

# TEST NORMAL
img_dir = folder + 'normal' + mode + '/'
os.system('ls ' + img_dir + ' > Image.txt')
name = '%s/%s_loss_normal_%s.txt' % (path,MODEL,mode)
f = open( name, "w" )
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        print( name )
        img = kimage.load_img(img_dir + name )
        x = np.array( img )
        x = x.reshape( 1, PATCH_SIZE, PATCH_SIZE, 3).astype('float32')
        x = (x - 127.5) / 127.5

        Ex = e_model.predict( x )
        GEx = g_model.predict( Ex )
        LG = tf.norm( x[0,:,:,:]-GEx[0,:,:,:], ord = 1)

        label = d_model.predict( {"img_input": x,  "z_input": Ex} )
        LD1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(label),logits=label))
        Ax1 = alpha * LG + (1.0 - alpha) * LD1

        fd1 = d_features.predict( {"img_input": x,  "z_input": Ex} )
        fd2 = d_features.predict( {"img_input": GEx,  "z_input": Ex} )
        LD2 = tf.norm( fd1 - fd2, ord = 1 )
        Ax2 = alpha * LG + (1.0 - alpha) * LD2

        f.write( '%f %f' % ( Ax1.numpy() , Ax2.numpy() ) )
        f.write( '\n' )
os.system('rm -r Image.txt')
f.close()

# TEST ANOMALY
img_dir = folder + 'anomaly' + mode + '/'
os.system('ls ' + img_dir + ' > Image.txt')
name = '%s/%s_loss_anomaly_%s.txt' % (path,MODEL,mode)
f = open( name, "w" )
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(img_dir + name)
        print( name )
        x = np.array( img )
        x = x.reshape( 1, PATCH_SIZE, PATCH_SIZE, 3).astype('float32')
        x = (x - 127.5) / 127.5

        Ex = e_model.predict( x )
        GEx = g_model.predict( Ex )
        LG = tf.norm( x[0,:,:,:]-GEx[0,:,:,:], ord = 1)

        label = d_model.predict( {"img_input": x,  "z_input": Ex} )
        LD1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(label),logits=label))
        Ax1 = alpha * LG + (1.0 - alpha) * LD1

        fd1 = d_features.predict( {"img_input": x,  "z_input": Ex} )
        fd2 = d_features.predict( {"img_input": GEx,  "z_input": Ex} )
        LD2 = tf.norm( fd1 - fd2, ord = 1 )
        Ax2 = alpha * LG + (1.0 - alpha) * LD2

        f.write( '%f %f' % ( Ax1.numpy() , Ax2.numpy() ) )
        f.write( '\n' )

os.system('rm -r Image.txt')
f.close()
