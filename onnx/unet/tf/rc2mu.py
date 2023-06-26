import os 
from os import listdir
from os.path import isfile, join
import glob
import pydicom
import pylab

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Dropout, Cropping3D
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.layers import concatenate 
   
class SpectXnet(object):
    def __init__(self, model_dir, model_file, input_dcm_dir, outpt_dcm_dir, slis, cols, rows):

        self.model_dir  = model_dir
        self.model_file = model_file
        self.input_dcm  = input_dcm_dir
        self.outpt_dcm  = outpt_dcm_dir
        self.slis = slis
        self.cols = cols
        self.rows = rows
        
    def unet(self):
        inputs = Input(shape=(self.slis, self.rows, self.cols, 1), dtype=tf.float32, name="spect_vol")

        conv1  = Conv3D(64,   3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1  = Conv3D(64,   3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1  = MaxPooling3D(pool_size=(1, 2, 2))(conv1)

        conv2  = Conv3D(128,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2  = Conv3D(128,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2  = MaxPooling3D(pool_size=(1, 2, 2))(conv2)

        conv3  = Conv3D(256,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3  = Conv3D(256,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3  = MaxPooling3D(pool_size=(1, 2, 2))(conv3)

        conv4  = Conv3D(512,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4  = Conv3D(512,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4  = Dropout(0.5)(conv4)
        pool4  = MaxPooling3D(pool_size=(1, 2, 2))(drop4)

        conv5  = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5  = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5  = Dropout(0.5)(conv5)

        up6    = Conv3D(512,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 4)
        conv6  = Conv3D(512,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6  = Conv3D(512,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7    = Conv3D(256,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 4)
        conv7  = Conv3D(256,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7  = Conv3D(256,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8    = Conv3D(128,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 4)
        conv8  = Conv3D(128,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8  = Conv3D(128,  3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9    = Conv3D(64,   3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (1,2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 4)
        conv9  = Conv3D(64,   3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9  = Conv3D(64,   3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9  = Conv3D(2,    3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv3D(1,    1, activation = 'sigmoid')(conv9)

        model = Model(inputs, conv10)
        model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model

    def load_model(self):
        file = os.path.join(self.model_dir, self.model_file)
        model = self.unet()
        model.load_weights(file)
        return model

    def save_model(self):
        net = self.load_model()
        net.save(self.model_dir)
        print(f"Saved model in {os.path.abspath(self.model_dir)}")
        
    def save_vols_tofile(self):
        filenames = [os.path.join(self.input_dcm, fn) for fn in os.listdir(self.input_dcm) if fn.endswith("IMA")]
        for filename in filenames:
            ds = pydicom.dcmread(filename)
            vol = ds.pixel_array # shape = (slice, column, row)
            with open(filename[:-4] + ".vol", 'wb') as f:
                for slice in vol:
                    slice.astype('float32').tofile(f)

    def test(self):
        print("\n===== testing =====")

        scale = 100_000

        inp_image = self.input_dcm
        out_mumap = self.outpt_dcm

        # load model
        print("\n--- load model ...")
        model = self.load_model()

        print("\n--- load dicom images ...")
        vNmFiles = [f for f in listdir(inp_image) if f.endswith(".IMA")]

        for idx, nm in enumerate(vNmFiles):

            # input/output files
            iFile = os.path.join(inp_image, nm)
            oFile = os.path.join(out_mumap, nm)
            
            print(f'\n{idx + 1}.')
            print(f'{"input" :<10}{iFile:>20}')

            iSeries = pydicom.dcmread(iFile)
            image  = iSeries.pixel_array
            max    = np.max(image)
            image  = image.astype('float32')/max

            size  = image.shape
            i_slis = size[0]
            i_cols = size[1]
            i_rows = size[2]
            print(f'{"volume shape " :<10}{size}')

            # infer from input DICOM
            oSeries = iSeries
            o_slis  = size[0]
            o_cols  = size[1]
            o_rows  = size[2]

            slis = i_slis;

            # for s in range(0, o_slis):
            #     oSeries.pixel_array[s][:] = 0
            for sl in oSeries.pixel_array:
                sl[:] = 0

            print('inference ...')
            for s in range(1, slis-1):
                input = image[s-1:s+2,:,:]
                input = input.reshape(1, 3, i_rows, i_cols, 1)
                outpt = model.predict(input, batch_size = 1, verbose=0)
                mumap = np.reshape(outpt, (3, o_cols, o_rows))
                slice = scale * mumap[1]
                frame = slice.reshape(o_rows, o_cols)
                oSeries.pixel_array[s][:] = frame

            print(f'{"output":<10}{oFile:>20}')
            oSeries.PixelData = oSeries.pixel_array.tobytes()
            oSeries.save_as(oFile)
                        
        print("\n====================")
  
    
if __name__ == '__main__':
   
    model_dir  = os.path.join(os.getcwd(), 'model')
    model_file = 'kr3sunet-bwhknz-adm-bce-acr-1b25e-model.hdf5'
    input_dcm  = '../data/ncRecon'
    outpt_dcm  = '../data/nnMumap'
    slis       = 3
    cols       = 128
    rows       = 128	

    model = SpectXnet(model_dir, model_file, input_dcm, outpt_dcm, slis, cols, rows)
    model.test()
    #model.save_model()
    #model.save_vols_tofile()
