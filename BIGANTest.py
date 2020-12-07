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

PATCH_SIZE = GO.PATCH_SIZE
latent_dim = GO.NOISE_DIM

if len(sys.argv) != 6:
    print('python3 BIGANTest.py ... \n')
    sys.exit( 1 )

# LOAD PARAMETERS
modelid = str( sys.argv[ 1 ] )
dataset = str( sys.argv[ 2 ] )
biganfolder = str( sys.argv[ 3 ] )
ofolder = str( sys.argv[ 4 ] )
oname = str( sys.argv[ 5 ] )

alpha = 0.9

# LOAD MODELS
e_model = MO.make_encoder_model( GO.IMAGE_DIM, GO.NOISE_DIM )
e_model.load_weights(  biganfolder + 'encoder_weights_' + modelid )
g_model = MO.make_generator_model( GO.NOISE_DIM )
g_model.load_weights( biganfolder + 'generator_weights_' + modelid )
d_model = MO.make_discriminator_model( GO.IMAGE_DIM, GO.NOISE_DIM )
d_model.load_weights( biganfolder + 'discriminator_weights_' + modelid )
d_features = tf.keras.Model( d_model.inputs, d_model.get_layer('dropout4').output )

os.system('mkdir ' + ofolder )
ofile = ofolder + "/" + oname
os.system('ls ' + dataset + ' > Image.txt')
f = open( ofile, "w" )
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        img = kimage.load_img(dataset + name )
        print( name )
        x = np.array( img )
        x = x.reshape( 1, PATCH_SIZE, PATCH_SIZE, 3).astype('float32')
        x = (x - 127.5) / 127.5

        Ex = e_model.predict( x )
        GEx = g_model.predict( Ex )
        LG = tf.norm( x[0,:,:,:] - GEx[0,:,:,:], ord = 1)
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
