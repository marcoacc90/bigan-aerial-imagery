import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import numpy as np
import os
import sys

if len(sys.argv) != 2:
    print('python3.6 PatchExtraction.py <img_dir>\n')
    print('python3.6 PatchExtraction.py Image');
    sys.exit( 1 )

PATCH_SIZE = 32
N_SAMPLES = 256

# CREATE DIRECTORIES
img_dir = str( sys.argv[ 1 ] )
os.system('ls ' + img_dir + ' > Image.txt')
os.system( 'mkdir normalTraining/');
nimage = 1
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        print(name)
        img = kimage.load_img( img_dir + name, target_size=(512, 512) )
        x = np.array( img )
        patch_image = simage.extract_patches_2d( x, (PATCH_SIZE,PATCH_SIZE), max_patches = N_SAMPLES )
        for i in range( N_SAMPLES ) :
            name = 'normalTraining/img_' + str( nimage ) + '_patch_' + str( i+1 ) + '.png';
            kimage.save_img( name, patch_image[ i, :, :, : ] )
        nimage = nimage + 1
os.system('rm -r Image.txt')
