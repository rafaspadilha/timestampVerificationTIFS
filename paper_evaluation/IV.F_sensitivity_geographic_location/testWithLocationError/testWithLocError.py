#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Script to evaluate models against samples with noise in location coordinates.
The model path and the modalities are passed as arguments to the script.

Usage:
    $ python testWithLocError.py model_path modality_id error_type location_error

Parameters:
    model_path: the path to the model hdf5 file
    modality_id: integer from [2, 4, 5]
        1 (not used): Ground-level image and timestamp
        2: Ground-level image, location, timestamp
        3 (not used): Ground-level image, sattelite image, timestamp
        4: All modalities
        5: All modalities + Transient Attribute Estimation
    error_type: if the error should be considered only on latitude, longitude or both
        "lat": latitude-only error
        "lon": longitude-only error
        "both": error in both coordinates
    location_error: float value with the absolute latitute and longitude degree of noise

Example:
    $ python testWithLocError.py ./gr_oh_loc_time_TA/weights.30-0.57407.hdf5 5 both 15
"""


#########################
# 		IMPORTS		    #
#########################

## General imports
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

## Dataloader
from dataLoader import DataLoaderWithLocError

## Keras
from tf.keras.models import Model, load_model
from tf.keras.losses import mean_squared_error



## GPU selection
import tensorflow as tf 
gpuNumber = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpuNumber], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpuNumber], True)



#########################
# 	   DEFINITIONS		#
#########################

# Path to the weights of the model that will be evaluated
pathToModel=sys.argv[1]

# Modality ID
modalityID=int(sys.argv[2])
assert modalityID in [2,4,5]

# Path to save the ROC files
pathToSaveRocs="./"

# Running multiple times (for different random tamperings)
## and we will report the average of such runs
nRuns=10

batchSize=32



## Amount and type of location error
errorType = sys.argv[3]
absoluteError = float(sys.argv[4])




#######################
##    Custom MSE     ##
#######################
# We will compute the MSE only for the consistent inputs
def transient_mse(y_true, y_pred):
    return tf.sum(mean_squared_error(y_true[0::2, :], y_pred[0::2, :]), axis = -1)




#######################
## Load architecture
#######################
model = load_model(pathToModel, custom_objects={"transient_mse":transient_mse})
print(model.summary())




#######################
## Testing Setup
#######################
includeSatellite=modalityID in [3, 4, 5]
outputTA=modalityID in [5]

dl = DataLoaderWithLocError(setToLoad="test",
                includeSatellite=includeSatellite, 
                outputTransientAttributes=outputTA)



print("-----------> Testing with ", absoluteError, " location noise")

accList = []
tTamperedRate = []
tRealRate = []
aucList = []

for runIdx in range(nRuns):
    yTrueList, yPredList, yScoreList = [], [], []
    for batch, labels in dl.loadTestDataInBatchesWithLocError(batchSize, errorType, absoluteError, seed=runIdx*42):
        if outputTA:
            # get only the consistOrNot Branch
            preds = model.predict_on_batch(batch)[0]
            # get only the consistOrNot Labels
            y_true = np.argmax(labels[0], axis = 1)
        else:
            preds=model.predict_on_batch(batch)
            y_true=np.argmax(labels, axis = 1)
            
        y_pred=np.argmax(preds, axis = 1)

        yTrueList += list(y_true)
        yPredList += list(y_pred)
        yScoreList += [p[1] for p in preds]


    acc = accuracy_score(yTrueList, yPredList)
    cm = confusion_matrix(yTrueList, yPredList)
    trr = cm[0,0] / float(np.sum(cm[0,:]))
    ttr = cm[1,1] / float(np.sum(cm[1,:]))

    accList += [acc]
    tTamperedRate += [ttr]
    tRealRate += [trr]

    fpr, tpr, _ = roc_curve(yTrueList, yScoreList)
    roc_auc = auc(fpr, tpr)
    aucList += [roc_auc]


### After all runs, print the average and std
print("==============")
print("ACC (mean +- std) = ", np.mean(accList), " +- ", np.std(accList))
print("True Real Rate (mean +- std) = ", np.mean(tRealRate), " +- ", np.std(tRealRate))
print("True Tampered Rate (mean +- std) = ", np.mean(tTamperedRate), " +- ", np.std(tTamperedRate))
print("AUC (mean +- std) = ", np.mean(aucList), " +- ", np.std(aucList), "\n\n\n")
