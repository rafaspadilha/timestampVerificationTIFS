#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Classes and methods used to load and preprocess the CVT dataset.
"""


#########################
# IMPORTS & DEFINITIONS #
#########################
import os
import h5py
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical 

#### Change the preprocessing function depending on the backbone architecture
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet


rootDatasetDir = "./CVT_dataset/"









##################################################
# Managing the paths for phone and aerial images #
##################################################

def get_full_ground_paths(short_ground_paths):
    phone_hf = os.path.join(rootDatasetDir, "ground_level/YFCC100M-PHONE/")
    amos_hf = os.path.join(rootDatasetDir, "ground_level/webcam-labeler/images/")
    ground_paths=[]
    for path in short_ground_paths:
        if not isinstance(path, str):
            path = path.decode("utf-8")

        if len(path.split("/"))>2:
            ground_paths.append(phone_hf+str(path))
        else:
            ground_paths.append(amos_hf+str(path))
    return ground_paths 


def get_aerial_paths(short_ground_paths):
    phone_hf = os.path.join(rootDatasetDir, "aerial/data/data/")
    amos_hf = os.path.join(rootDatasetDir, "aerial/data/data/aerial_amos_18/")
    aerial_paths=[]
    for path in short_ground_paths:
        if not isinstance(path, str):
            path = path.decode("utf-8")
        
        if len(path.split("/"))>2: # phone_images     
            aerial_paths.append(phone_hf+path.replace("images", "aerial_18"))
        else: #amos images
            aerial_paths.append(amos_hf+path.split("/")[0]+"/"+path.split("/")[0]+".jpg")

    return aerial_paths








##################################################
#       Input data preprocessing methods         #
##################################################

def load_preprocess_aerialImg(im_path, sz=[224,224]):
    im = load_img(im_path)
    im = im.resize(sz)
    im = img_to_array(im)
    im = preprocess_densenet(im)
    return im

def load_preprocess_groundImg(im_path, sz=[224,224]):
    im = load_img(im_path)
    im = im.resize(sz)
    im = img_to_array(im)
    im = preprocess_densenet(im) 
    return im


#### Convert latitude and longitude to ECEF
def preprocess_loc(location, alt=0):
    lat, lon = location
    _earth_radius = 6378137.0
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    rad = np.float64(_earth_radius)      # Radius of the Earth (in meters)
    f = np.float64(1.0 / 298.257223563)  # Flattening factor WGS84 Model

    cosLat = np.cos(lat)
    sinLat = np.sin(lat)

    FF = (1.0 - f)**2
    C = 1.0 / np.sqrt(cosLat**2 + FF * sinLat**2)
    S = C*FF

    x = ((rad * C + alt) * cosLat * np.cos(lon)) / _earth_radius
    y = ((rad * C + alt) * cosLat * np.sin(lon)) / _earth_radius
    z = ((rad * S + alt) * sinLat) / _earth_radius

    return np.array([x,y,z])


#### Convert month and hour to [-1, 1]
def preprocess_time(time):
    month, hour = time

    month = 2.0 * ((month - 1) / 11.0 - 0.5)
    hour = 2.0 * (hour/23.0 - 0.5)

    return np.array([month, hour])








##################################################
#               DataLoader class                 #  
##################################################

class DataLoader(object):
    def __init__(self, setToLoad="train", includeLocation=True, includeSatellite=True, outputTransientAttributes=True):
        assert setToLoad in ["train", "test"]

        self.set = setToLoad
        self.includeLocation = includeLocation
        self.includeSatellite = includeSatellite
        self.outputTA = outputTransientAttributes
        self.setup()

    #### Setup paths and preload dataset file
    def setup(self):
        if self.set == "train":
            h5FileName = os.path.join(rootDatasetDir, "train_combine.h5")
        else:
            h5FileName = os.path.join(rootDatasetDir, "test_combine.h5")


        with h5py.File(h5FileName, 'r') as hf:
            self.locLabels = list(zip(hf.get("lats")[:], hf.get("lons")[:]))
            self.groundPaths = get_full_ground_paths(hf.get("paths")[:])
            self.aerialPaths = get_aerial_paths(hf.get("paths")[:])

            times = [i.decode() for i in hf.get("time")[:]]
            months = np.array([int(time.split("-")[1]) for time in times])
            hours = np.array([ int(time.split("-")[2].split(":")[0].split(" ")[-1]) for time in times])
            self.timeLabels = np.concatenate((months[:,np.newaxis],hours[:,np.newaxis]),axis=1)

            self.transientAttributes = hf.get("trans_preds")[:]

        self.nPairsInSet = len(self.groundPaths) 
        self.setSize = 2*self.nPairsInSet #2* since half of the batch will be with tampered imgs



    #### Method that generates a timestamp manipulation
    # It randomly picks a hour/month from another image in the set
    # with a different hour/month to the consistent sample
    def fakeTime(self, time):
        month, hour = time

        fakeMonth, fakeHour = random.choice(self.timeLabels)
        while (month == fakeMonth) and (hour == fakeHour):
            fakeMonth, fakeHour = random.choice(self.timeLabels)

        return preprocess_time((fakeMonth, fakeHour))





    ##################################################
    #               Loader methods                   #  
    ##################################################
    """
    Both methods bellow load batches of data and correspondent labels
    - loadImagesInBatches: loads a continuous stream of data for 
    training shuffling it at the end of an epoch. 
    - loadTestDataInBatches: loads the set only once. We used it 
    for testing/evaluation of the models.
    """
    
    def loadImagesInBatches(self, batchSize):
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
                labels += [to_categorical(0, num_classes=2), to_categorical(1, num_classes=2)]

                # Add the transient attributes to the output
                if self.outputTA:
                    transAtt += [self.transientAttributes[idx], self.transientAttributes[idx]]

                nInBatch += 2
                if nInBatch >= batchSize:
                    inputBatch = [np.array(gBatch)]
                    inputBatch += [np.array(aBatch)] if self.includeSatellite else []
                    inputBatch += [np.array(lBatch)] if self.includeLocation else []
                    inputBatch += [np.array(tBatch)]

                    outputBatch = [np.array(labels)]
                    outputBatch += [np.array(transAtt), np.array(transAtt)] if self.outputTA else []

                    yield inputBatch, outputBatch
                    gBatch, aBatch, tBatch, lBatch, labels, transAtt = [], [], [], [], [], []
                    inputBatch, outputBatch = [], []
                    nInBatch = 0




    # For evaluation, the seed controls the randomness of the timestamp manipulation
    # to make sure all experiments are done with the same set
    def loadTestDataInBatches(self, batchSize, seed=42):
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
    ld = DataLoader("test")

    for batch, gt in ld.loadTestDataInBatches(1):
        print(len(batch), [x.shape for x in batch])
        print(len(gt), [x.shape for x in gt])
        print(gt[1][0])
        break
