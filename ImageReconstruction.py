import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import Modules.Model as MO
import Modules.Tools as TO
import Modules.Global as GO
import os
import sys

# Generate the reconstructed images by
    # AUTOENCODER
    # GANIZIf
PATCH_SIZE = GO.PATCH_SIZE
latent_dim = GO.NOISE_DIM

def loadDataset( dataset ) :
    count = 0
    os.system('ls ' + dataset + ' > Image.txt')
    flag = False
    with open('Image.txt', 'r') as filehandle:
        for line in filehandle:
            name = line[:-1]
            img = kimage.load_img(dataset + name)
            x = np.array( img )
            x = x.reshape( (1,PATCH_SIZE,PATCH_SIZE,3) )
            if flag == True:
                test_images = np.concatenate( (test_images,x), axis = 0)
            else:
                test_images = np.copy( x )
                flag = True
            count = count + 1
            if count == 1000 :
                break;
    os.system('rm -r Image.txt')
    test_images = test_images.astype('float32')
    test_images = (test_images - 127.5) / 127.5
    return test_images

def main():
    modelid = str( 10000 )
    dataset = 'Dataset3/'
    ofolder = 'Reconstruction/'

    biganfolder = 'BIGANTest_Models/'

    # LOAD MODELS
    e_model = MO.make_encoder_model( GO.IMAGE_DIM, latent_dim )
    g_model = MO.make_generator_model( latent_dim )

    g_model.load_weights( biganfolder + 'generator_weights_' + modelid )
    e_model.load_weights( biganfolder  + 'encoder_weights_' + modelid )

    # CREATE FOLDERS
    os.system( 'mkdir ' + ofolder )

    # LOAD DATASET
    x_normal = loadDataset( dataset + 'normalTest/' )
    x_novel = loadDataset( dataset  + 'novelTest/' )

    # SAVE
    nsol = 10
    n_samples = 100
    for i in range( nsol ) :
        x,_ = TO.generateRealSamples( x_normal, n_samples )
        y,_ = TO.generateRealSamples( x_novel, n_samples )

        Ex = e_model.predict( x )
        GEx = g_model.predict( Ex )

        Ey = e_model.predict( y )
        GEy = g_model.predict( Ey )

        # Real images
        x = TO.conver2image( x )
        TO.savePlot( ofolder + str(i) + '_normal.png', x )
        y = TO.conver2image( y )
        TO.savePlot( ofolder + str(i) + '_novel.png', y )

        # Reconstructed
        GEx = TO.conver2image( GEx )
        TO.savePlot( ofolder + str(i) + '_normal_' + 'bigan' + '.png', GEx )
        GEy = TO.conver2image( GEy )
        TO.savePlot( ofolder + str(i) + '_novel_' + 'bigan' + '.png', GEy )


if __name__ == "__main__":
    main()
