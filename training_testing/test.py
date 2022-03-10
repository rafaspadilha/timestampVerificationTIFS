#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Example of testing script considering DenseNet as backbone,
location and satellite included as input modalities, and
multi-task optimization (including transient attribute estimation)
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
# pathToModel = sys.argv[1] ### uncomment to pass it as parameter
pathToModel = "../models/dummy_model.hdf5"

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
dl = DataLoader(setToLoad="test", includeLocation=True, 
                includeSatellite=True, outputTransientAttributes=True)


### List to store the statistics for each run
accList = []
tTamperedRate = []
tRealRate = []
aucList = []

for runIdx in range(nRuns):
	print("\n\nRun --> ", runIdx+1, " / ", nRuns)
	yTrueList, yPredList, yScoreList = [], [], []
	for batch, labels in dl.loadTestDataInBatches(batchSize, seed=runIdx*42):
		preds = model.predict_on_batch(batch)[0] #get only the consistOrNot Branch

		y_true = np.argmax(labels[0], axis=1) #get only the consistOrNot Labels
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
