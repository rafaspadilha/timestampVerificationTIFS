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
sys.path.append("../datasets")

import numpy as np

from dataLoader import *


#### Convert month and hour to [-1, 1]
def preprocess_time(time):
    month, hour = time

    month = 2.0 * ((month - 1) / 11.0 - 0.5)
    hour = 2.0 * (hour/23.0 - 0.5)

    return np.array([month, hour])


class DataLoaderSat(DataLoader):
    def __init__(self, setToLoad="train"):
        DataLoader.__init__(self, setToLoad,
                            includeLocation = True,
                            includeSatellite = True,
                            outputTransientAttributes = True)

    def loadSatImagesInBatches(self, batchSize):
            aBatch, lBatch = [], []
            nInBatch = 0

            aerialAndLoc = list(zip(self.aerialPaths, self.locLabels))
            uniqueAerialAndLoc = list(set(aerialAndLoc))

            for idx in range(len(uniqueAerialAndLoc)):
                    try:
                        aImg = load_preprocess_aerialImg(
                            uniqueAerialAndLoc[idx][0])
                    except (OSError, IOError) as e:
                        #this might happen if it can't read the img
                        continue

                    aBatch += [aImg]

                    loc = uniqueAerialAndLoc[idx][1]
                    lBatch += [loc]

                    nInBatch += 1
                    if nInBatch >= batchSize:
                        yield np.array(aBatch), lBatch
                        aBatch, lBatch = [], []
                        nInBatch = 0

            #Yield the final batch, if smaller than batchSize
            if nInBatch > 0:
                yield np.array(aBatch), lBatch
