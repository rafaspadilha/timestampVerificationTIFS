#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
import numpy as np
import random
"""
Extending the dataLoader class, focusing on the Location Errors
"""

import sys
sys.path.append("../../../datasets")
from dataLoader import *




def randomPlusOrMinus():
    return 1 if random.random() < 0.5 else -1




class DataLoaderWithLocError(DataLoader):
    def __init__(self, setToLoad="train", includeSatellite=True, outputTransientAttributes=True):
        DataLoader.__init__(self, setToLoad,
                            includeLocation=True,
                            includeSatellite=includeSatellite,
                            outputTransientAttributes=outputTransientAttributes)








    #### Method used to generate batches of data with Location Augmentation
    def loadImagesInBatchesWithLocAug(self, batchSize):
        # Initialize arrays to store each input modality and labels
        inputBatch, outputBatch = [], []
        gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []

        # Count the number of samples in the batch
        nInBatch = 0

        ### Epoch loop
        while 1:
            ## Shuffle data at each epoch
            rndIdx = np.arange(self.nPairsInSet)
            np.random.shuffle(rndIdx)

            for idx in rndIdx:

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

                if self.includeSatellite:
                    aBatch += [aImg, aImg]





                #Augment location
                originalLoc = np.array(self.locLabels[idx])

                if random.choice([0, 1]) == 1:
                    errorPercentage = random.choice(
                        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
                    originalLoc[0] *= 1 + (errorPercentage * randomPlusOrMinus())
                    originalLoc[0] = max(-90.0, min(originalLoc[0], 90.0))

                if random.choice([0, 1]) == 1:
                    errorPercentage = random.choice(
                        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
                    originalLoc[1] *= 1 + (errorPercentage * randomPlusOrMinus())
                    originalLoc[1] = max(-180.0, min(originalLoc[1], 180.0))

                #Add the location information to the batch
                loc = preprocess_loc(originalLoc)
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

                    yield inputBatch, outputBatch
                    gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []
                    inputBatch, outputBatch = [], []
                    nInBatch = 0




    def loadTestDataInBatchesWithLocError(self, batchSize, errorType, absoluteError, seed=42):
        # Initialize arrays to store each input modality and labels
        inputBatch, outputBatch = [], []
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

                if self.includeSatellite:
                    aBatch += [aImg, aImg]






                ### Perturbing the location
                originalLoc = list(self.locLabels[idx])
                

                if errorType == "lat":
                    originalLoc[0] += (absoluteError * randomPlusOrMinus())
                    originalLoc[0] = max(-90.0, min(originalLoc[0], 90.0))
                    
                elif errorType == "lon":
                    originalLoc[1] += (absoluteError * randomPlusOrMinus())
                    originalLoc[1] = max(-180.0, min(originalLoc[1], 180.0))

                elif errorType == "both":
                    originalLoc[0] += (absoluteError * randomPlusOrMinus())
                    originalLoc[0] = max(-90.0, min(originalLoc[0], 90.0))

                    originalLoc[1] += (absoluteError * randomPlusOrMinus())
                    originalLoc[1] = max(-180.0, min(originalLoc[1], 180.0))
                else:
                    raise Exception()

                #Add the location information to the batch
                loc = preprocess_loc(originalLoc)
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

                    yield inputBatch, outputBatch
                    gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []
                    inputBatch, outputBatch = [], []
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

            yield inputBatch, outputBatch


##################################################
#               Sanity Check                     #
##################################################
if __name__ == '__main__':
    ld = DataLoaderWithLocError("test")

    for batch, gt in ld.loadTestDataInBatches(1, "both", 0.1):
        print(len(batch), [x.shape for x in batch])
        print(len(gt), [x.shape for x in gt])
        print(gt[1][0])

        break
