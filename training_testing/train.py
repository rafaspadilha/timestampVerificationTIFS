#####################################################
# Content-Aware Detection of Timestamp Manipulation #
# IEEE Trans. on Information Forensics and Security #
# R. Padilha, T. Salem, S. Workman,                 #
# F. A. Andalo, A. Rocha, N. Jacobs                 #
#####################################################

##### DESCRIPTION
"""
Example of training script considering DenseNet as backbone,
location and satellite included as input modalities, and
multi-task optimization (including transient attribute estimation)
"""


#########################
# IMPORTS & DEFINITIONS #
#########################
import os, sys

## Dataloader
sys.path.append("../datasets")
from dataLoader import DataLoader

## Architectures
sys.path.append("../architectures")
from model_DenseNet import buildModel


## Training-related imports
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


## GPU selection
import tensorflow as tf 
gpuNumber = 3
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpuNumber], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpuNumber], False)


## Training definitions
batchSize = 32
nEpochs = 30
regTerm = 0.01
pathToSaveCheckpoints = "./models/"



#######################
##    Custom MSE     ##
#######################
# We will compute the MSE only for the consistent inputs
def transient_mse(y_true, y_pred):
    return tf.sum(mean_squared_error(y_true[0::2,:], y_pred[0::2,:]), axis=-1)


#######################
## Build Architecture
#######################
model = buildModel(includeLocation=True, includeSatellite=True, 
                    outputTransientAttributes=True, regTerm=0.01)
print(model.summary())


#######################
## Training Setup
#######################
dl = DataLoader(setToLoad="train", includeLocation=True, 
                includeSatellite=True, outputTransientAttributes=True)

trainPairs = dl.setSize
trainBatchesPerEpoch = int(trainPairs/batchSize)

print("-----------> Training Setup")
print("Number of Epochs = ", nEpochs)
print("L2 Regularization = ", regTerm)
print("Batch Size = ", batchSize)
print("Train Batches per Epoch = ", trainBatchesPerEpoch)


# Create dirs to save checkpoints and logs
if not os.path.exists(pathToSaveCheckpoints):
	os.makedirs(pathToSaveCheckpoints)

# Instantiate callbacks
checkpointer = ModelCheckpoint(filepath=os.path.join(pathToSaveCheckpoints,"weights.{epoch:02d}-{loss:.5f}.hdf5"), verbose=0)

# Compile model
opt = Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss={"consist_fc3":"categorical_crossentropy", 
                                    "gr_trans_fc3":transient_mse, 
                                    "ae_loc_time_trans_fc3":transient_mse}, 
                            metrics={"consist_fc3":"accuracy", 
                                    "gr_trans_fc3":"mae", 
                                    "ae_loc_time_trans_fc3":"mae"})




#######################
## Training
#######################
model.fit_generator(dl.loadImagesInBatches(batchSize),
                    steps_per_epoch=trainBatchesPerEpoch, epochs=nEpochs,
                    verbose=1, callbacks=[checkpointer])

