import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.feature_extraction import image as simage
import numpy as np
import os
import sys

if len(sys.argv) != 3:
    print('python3.6 PatchExtractionTest.py <img_dir> <map_dir>\n')
    print('python3.6 PatchExtractionTest.py Image Reference');
    sys.exit( 1 )

PATCH_SIZE = 32
N_SAMPLES = 256
TH = int( 0.8 * (PATCH_SIZE * PATCH_SIZE) )


# CREATE DIRECTORIES
img_dir = str( sys.argv[ 1 ] )
map_dir = str( sys.argv[ 2 ] )
os.system('ls ' + img_dir + ' > Image.txt')

os.system('mkdir normalTest/' )
os.system('mkdir anomalyTest/' )

nimage = 1
myseed = int( time.time() );
with open('Image.txt', 'r') as filehandle:
    for line in filehandle:
        name = line[:-1]
        print(name)
        img = kimage.load_img( img_dir + name, target_size=(512, 512) )
        map = kimage.load_img( map_dir + name, color_mode='grayscale', target_size=(512, 512) )
        x = np.array( img )
        y = np.array( map )
        patch_image = simage.extract_patches_2d( x, (PATCH_SIZE,PATCH_SIZE), max_patches = N_SAMPLES, random_state = myseed)
        patch_map = simage.extract_patches_2d( y, (PATCH_SIZE,PATCH_SIZE), max_patches = N_SAMPLES, random_state = myseed)
        for i in range( N_SAMPLES ) :
            a = np.sum( patch_map[ i, :, : ] ) / 255.0;
            if a == 0.0 :  # Normal image
                name = 'normalTest/img_' + str( nimage ) + '_patch_' + str( i+1 ) + '.png';
                kimage.save_img( name, patch_image[ i, :, :, : ] )
            elif a >= TH :   # Novel
                name = 'anomalyTest/img_' + str( nimage ) + '_patch_' + str( i+1 ) + '.png';
                kimage.save_img( name, patch_image[ i, :, :, : ] )
        nimage = nimage + 1
        myseed = myseed + 1
os.system('rm -r Image.txt')
