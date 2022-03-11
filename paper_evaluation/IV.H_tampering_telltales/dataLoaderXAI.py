#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Extending the dataLoader class, focusing on the Satellite images
"""

import sys
sys.path.append("../../datasets")

import numpy as np
import random

from dataLoader import *


#### Convert month and hour to [-1, 1]
def preprocess_time(time):
    month, hour = time

    month = 2.0 * ((month - 1) / 11.0 - 0.5)
    hour = 2.0 * (hour/23.0 - 0.5)

    return np.array([month, hour])


class DataLoaderXAI(DataLoader):
    def __init__(self, setToLoad="train"):
        DataLoader.__init__(self, setToLoad,
                            includeLocation = True,
                            includeSatellite = True,
                            outputTransientAttributes = True)

    # For evaluation, the seed controls the randomness of the timestamp manipulation
    # to make sure all experiments are done with the same set
    def loadTestDataInBatchesWithPath(self, batchSize, seed=42):
        # Initialize arrays to store each input modality and labels
        inputBatch, outputBatch, pathBatch = [], [], []
        gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []

        # Count the number of samples in the batch
        nInBatch = 0

        # Set the seed
        random.seed(seed)

        idxList = range(len(self.groundPaths))
        for idx in idxList:

                # Load and preprocess ground-level image
                try:
                    gImg = load_preprocess_groundImg(self.groundPaths[idx])
                except (OSError, IOError) as e:
                    #If error in loading, go to the next sample
                    continue

                # Load and preprocess satellite image
                if self.includeSatellite:
                    try:
                        aImg = load_preprocess_aerialImg(self.aerialPaths[idx])
                    except (OSError, IOError) as e:
                        #If error in loading, go to the next sample
                        continue

                # Include each image in the batch twice (consistent and inconsistent)
                gBatch += [gImg, gImg]

                # Add the path
                pathBatch += [self.groundPaths[idx]]

                if self.includeSatellite:
                    aBatch += [aImg, aImg]

                #Add the location information to the batch
                if self.includeLocation:
                    loc = preprocess_loc(self.locLabels[idx])
                    lBatch += [loc, loc]

                #Process time info and tamper the time for one pair
                time = preprocess_time(self.timeLabels[idx])
                fakeTime = self.fakeTime(self.timeLabels[idx])
                tBatch += [time, fakeTime]

                #Label 0 = real/consistent tuple
                #Label 1 = tampered/inconsistent tuple
                labels += [to_categorical(0, num_classes=2),
                           to_categorical(1, num_classes=2)]

                # Add the transient attributes to the output
                if self.outputTA:
                    transAtt += [self.transientAttributes[idx],
                                 self.transientAttributes[idx]]

                nInBatch += 2
                if nInBatch >= batchSize:
                    inputBatch = [np.array(gBatch)]
                    inputBatch += [np.array(aBatch)
                                   ] if self.includeSatellite else []
                    inputBatch += [np.array(lBatch)
                                   ] if self.includeLocation else []
                    inputBatch += [np.array(tBatch)]

                    outputBatch = [np.array(labels)]
                    outputBatch += [np.array(transAtt),
                                    np.array(transAtt)] if self.outputTA else []

                    yield inputBatch, pathBatch, outputBatch
                    gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []
                    inputBatch, pathBatch, outputBatch = [], [], []
                    nInBatch = 0

        #Yield the final batch, if smaller than batchSize
        if nInBatch > 0:
            inputBatch = [np.array(gBatch)]
            inputBatch += [np.array(aBatch)
                           ] if self.includeSatellite else []
            inputBatch += [np.array(lBatch)
                           ] if self.includeLocation else []
            inputBatch += [np.array(tBatch)]

            outputBatch = [np.array(labels)]
            outputBatch += [np.array(transAtt),
                            np.array(transAtt)] if self.outputTA else []

            yield inputBatch, pathBatch, outputBatch


    def loadTestDataInBatchesWithPath(self, batchSize, allTestSet=False, seed=42):
        gBatch, aBatch, tBatch, lBatch, labels = [], [], [], [], []
        pathBatch = []
        nInBatch = 0
        random.seed(seed)
        #To repeat Twafiq's experiment, we will only use the first 2k pairs
        if allTestSet:
            idxList = range(len(self.groundPaths))
        else:
            idxList = range(2000)

        for idx in idxList:
                try:
                    gImg = load_preprocess_groundImg(self.groundPaths[idx])
                    aImg = load_preprocess_aerialImg(self.aerialPaths[idx])
                except (OSError, IOError) as e:
                    #this might happen if it can't read the img
                    continue

                pathBatch += [self.groundPaths[idx]]

                gBatch += [gImg, gImg]
                aBatch += [aImg, aImg]

                #Add the location information to the batch
                loc = preprocess_loc(self.locLabels[idx])
                lBatch += [loc, loc]

                #Process time info and fake the time for one pair
                time = preprocess_time(self.timeLabels[idx])
                fakeTime = self.fakeTime(self.timeLabels[idx])
                tBatch += [time, fakeTime]

                #Label 0 = real/consistent tuple
                #Label 1 = tampered/inconsistent tuple
                labels += [to_categorical(0, num_classes=2),
                           to_categorical(1, num_classes=2)]

                nInBatch += 2
                if nInBatch >= batchSize:
                    yield [np.array(gBatch), np.array(aBatch), np.array(lBatch), np.array(tBatch)], pathBatch, np.array(labels)
                    gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []
                    pathBatch = []
                    nInBatch = 0

        #Yield the final batch, if smaller than batchSize
        if nInBatch > 0:
            yield [np.array(gBatch), np.array(aBatch), np.array(lBatch), np.array(tBatch)], pathBatch, np.array(labels)
