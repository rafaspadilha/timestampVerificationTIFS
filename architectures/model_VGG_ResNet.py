#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Model definition for the architecture optimized for consistency
verification and transient attribute estimation
"""

#########################
# IMPORTS & DEFINITIONS #
#########################
import numpy as np
import os


## Architectures

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16

## Keras layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.regularizers import l2




## Auxiliary method for renaming the layers of both visual encoders
# in case they use the same architecture (e.g., two denseNets)
# this is needed as both encoders do not share weights
def renameModel(model, prefix):
    for layer in model.layers:
        layer._name = prefix + layer.name
    model.name = prefix + model.name


##############################################
######## Build Model method
# Parameters:
# - includeLocation: if location coordinates are an input to the model (boolean)
# - includeSatellite: if aerial image is an input to the model (boolean)
# - outputTransientAttributes: if the model will estimate the transient attributes (boolean)
# - regTerm: L2 regularization term used in most fully-connected layers

#######################

def buildModel(includeLocation=True, includeSatellite=True, outputTransientAttributes=True,
               regTerm=0.01):

    inputList = []
    outputList = []

    ########################### Ground-level Image
    groundInput = Input(shape=(224,224,3), name='groundInput')
    groundBaseModel = VGG16(
        include_top=False, weights='imagenet', input_tensor=groundInput, pooling=None)
    renameModel(groundBaseModel, prefix="gr_")
    groundFeatures = groundBaseModel(groundInput)
    groundFeatures = Flatten()(groundFeatures)
    groundFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='ground_fc1')(groundFeatures)
    groundFeatures=  BatchNormalization()(groundFeatures)
    groundFeatures = Dense(128, activation='relu', kernel_regularizer=l2(regTerm), name='ground_fc2')(groundFeatures)
    groundFeatures=  BatchNormalization()(groundFeatures)
    ###########################
    
    ########################### Time
    timeInput = Input(shape=(2,), name='timeInput')
    timeFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='time_fc1')(timeInput)
    timeFeatures = BatchNormalization()(timeFeatures)
    timeFeatures = Dense(512, activation='relu', kernel_regularizer=l2(regTerm), name='time_fc2')(timeFeatures)
    timeFeatures = BatchNormalization()(timeFeatures)
    timeFeatures = Dense(128, activation='relu', kernel_regularizer=l2(regTerm), name='time_fc3')(timeFeatures)
    timeFeatures = BatchNormalization()(timeFeatures)
    ###########################




    ########################### Satellite Image
    if includeSatellite:
        aerialInput = Input(shape=(224,224,3), name="aerialInput")
        aerialBaseModel = ResNet50V2(include_top=False, weights='imagenet', input_tensor=aerialInput, pooling=None)
        renameModel(aerialBaseModel, prefix="ae_")
        aerialFeatures = aerialBaseModel(aerialInput)
        aerialFeatures = Flatten()(aerialFeatures)
        aerialFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='aerial_fc1')(aerialFeatures)
        aerialFeatures=  BatchNormalization()(aerialFeatures)
        aerialFeatures = Dense(128, activation='relu', kernel_regularizer=l2(regTerm), name='aerial_fc2')(aerialFeatures)
        aerialFeatures=  BatchNormalization()(aerialFeatures)
    ###########################




    ########################### Location Coordinates
    if includeLocation:
        locationInput = Input(shape=(3,), name='locationInput')
        locFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='loc_fc1')(locationInput)
        locFeatures=  BatchNormalization()(locFeatures)
        locFeatures = Dense(512, activation='relu', kernel_regularizer=l2(regTerm), name='loc_fc2')(locFeatures)
        locFeatures=  BatchNormalization()(locFeatures)
        locFeatures = Dense(128, activation='relu', kernel_regularizer=l2(regTerm), name='loc_fc3')(locFeatures)
        locFeatures=  BatchNormalization()(locFeatures)
    ###########################

    




    ## Concatenate the features and set the inputList
    listOfFeatures = [groundFeatures]
    inputList += [groundInput]

    if includeSatellite:
        listOfFeatures += [aerialFeatures]
        inputList += [aerialInput]

    if includeLocation:
        listOfFeatures += [locFeatures]
        inputList += [locationInput]

    listOfFeatures += [timeFeatures]
    inputList += [timeInput]

    combinedFeatures = Concatenate(axis=-1)(listOfFeatures)





    ######## Consistency Verification Branch
    consistFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='consist_fc1')(combinedFeatures)
    consistFeatures=  BatchNormalization()(consistFeatures)
    consistFeatures = Dense(512, activation='relu', kernel_regularizer=l2(regTerm), name='consist_fc2')(consistFeatures)
    consistFeatures=  BatchNormalization()(consistFeatures)
    consistPred = Dense(2, activation='softmax', kernel_regularizer=l2(regTerm), name='consist_fc3')(consistFeatures)

    outputList += [consistPred]
    ###########################






    ######### Transient Attributes Branch
    if outputTransientAttributes:

        #### aG - Estimate only from Ground-Level Image
        grTransFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='gr_trans_fc1')(groundFeatures)
        grTransFeatures=  BatchNormalization()(grTransFeatures)
        grTransFeatures = Dense(512, activation='relu', kernel_regularizer=l2(regTerm), name='gr_trans_fc2')(grTransFeatures)
        grTransFeatures=  BatchNormalization()(grTransFeatures)
        transGRPred = Dense(40, activation='sigmoid', kernel_regularizer=l2(regTerm), name='gr_trans_fc3')(grTransFeatures)

        #### aG - Estimate from Time + Aerial Img + Location
        transCombinedFeatures = Concatenate(axis=-1)(listOfFeatures[1:]) #Ignore ground-level image features
        transCombinedFeatures = Dense(256, activation='relu', kernel_regularizer=l2(regTerm), name='ae_loc_time_trans_fc1')(transCombinedFeatures)
        transCombinedFeatures =  BatchNormalization()(transCombinedFeatures)
        transCombinedFeatures = Dense(512, activation='relu', kernel_regularizer=l2(regTerm), name='ae_loc_time_trans_fc2')(transCombinedFeatures)
        transCombinedFeatures =  BatchNormalization()(transCombinedFeatures)
        transCombinedPred = Dense(40, activation='sigmoid', kernel_regularizer=l2(regTerm), name='ae_loc_time_trans_fc3')(transCombinedFeatures)

        ## Include the TAs in the outputList
        outputList += [transGRPred, transCombinedPred]




    return Model(inputs=inputList, outputs=outputList)
