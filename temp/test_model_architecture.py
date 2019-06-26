from keras.models import load_model
import os
import segmentation_models

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

modelfile='/home/omnisky/PycharmProjects/data/models/GF2/GF2_unet_vgg16_categorical_crossentropy_480_2019-06-20_08-26-16test.h5'

model=load_model(modelfile)
print(model.summary())