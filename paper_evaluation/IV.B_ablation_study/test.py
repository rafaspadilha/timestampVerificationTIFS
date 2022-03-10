#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Testing script considering DenseNet as backbone.
The model path and the modalities are passed as arguments to the script.

Usage:
	$ python test.py model_path modality_id

Parameters:
	model_path: the path to the model hdf5 file
	modality_id: integer from 1 to 5
		1: Ground-level image and timestamp
		2: Ground-level image, location, timestamp
		3: Ground-level image, sattelite image, timestamp
		4: All modalities
		5: All modalities + Transient Attribute Estimation

Example:
	$ python test.py ./gr_oh_loc_time_TA/weights.30-0.57407.hdf5 5
"""


#########################
# 		IMPORTS		    #
#########################

## General imports
import numpy as np
import os, sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

## Dataloader
sys.path.append("../datasets")
from dataLoader import DataLoader


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
pathToModel = sys.argv[1]

# Modality ID
modalityID = int(sys.argv[2])

# Path to save the ROC files
pathToSaveRocs = "./"

# Running multiple times (for different random tamperings)
## and we will report the average of such runs
nRuns = 10

batchSize = 32


#######################
##    Custom MSE     ##
#######################
# We will compute the MSE only for the consistent inputs
def transient_mse(y_true, y_pred):
    return tf.sum(mean_squared_error(y_true[0::2,:], y_pred[0::2,:]), axis=-1)


#######################
## Load architecture
#######################
model = load_model(pathToModel, custom_objects={"transient_mse":transient_mse})
print(model.summary())



#######################
## Testing Setup
#######################
includeLocation = modalityID in [2, 4, 5]
includeSatellite = modalityID in [3, 4, 5]
outputTA = modalityID in [5]

dl = DataLoader(setToLoad="test", 
				includeLocation=includeLocation, 
                includeSatellite=includeSatellite, 
				outputTransientAttributes=outputTA)


### List to store the statistics for each run
accList = []
tTamperedRate = []
tRealRate = []
aucList = []

for runIdx in range(nRuns):
	print("\n\nRun --> ", runIdx+1, " / ", nRuns)
	yTrueList, yPredList, yScoreList = [], [], []
	for batch, labels in dl.loadTestDataInBatches(batchSize, seed=runIdx*42):

		if outputTA:
			preds = model.predict_on_batch(batch)[0] # get only the consistOrNot Branch
			y_true = np.argmax(labels[0], axis=1)  # get only the consistOrNot Labels
		else:
			preds = model.predict_on_batch(batch)
			y_true = np.argmax(labels, axis=1)

		
		y_pred = np.argmax(preds, axis=1)

		yTrueList += list(y_true)
		yPredList += list(y_pred)
		yScoreList += [p[1] for p in preds]


	acc = accuracy_score(yTrueList, yPredList)
	cm = confusion_matrix(yTrueList, yPredList)
	trr = cm[0,0] / float(np.sum(cm[0,:]))
	ttr = cm[1,1] / float(np.sum(cm[1,:]))

	print("Acc = ", acc)
	print("True Real Rate = ", trr)
	print("True Tampered Rate = ", ttr)
	print("Conf Matrix")
	print(cm)

	accList += [acc]
	tTamperedRate += [ttr]
	tRealRate += [trr]

	fpr, tpr, _ = roc_curve(yTrueList, yScoreList)
	roc_auc = auc(fpr, tpr)

	aucList += [roc_auc]
	print("AUC = ", roc_auc)

	print(fpr.shape, tpr.shape, np.array([fpr, tpr]).shape)
	np.save(os.path.join(pathToSaveRocs,"fpr_tpr_run_" + str(runIdx)), np.array([fpr, tpr]))




### After all runs, print the average and std
print("==============")
print("ACC (mean +- std) = ", np.mean(accList), " +- ", np.std(accList))
print("True Tampered Rate (mean +- std) = ", np.mean(tTamperedRate), " +- ", np.std(tTamperedRate))
print("True Real Rate (mean +- std) = ", np.mean(tRealRate), " +- ", np.std(tRealRate))
print("AUC (mean +- std) = ", np.mean(aucList), " +- ", np.std(aucList))
