# import the necessary packages
import torch
import os

# base path of the dataset
OUTPUT_PATH = "output"
INPUT_PATH = os.path.join(OUTPUT_PATH, "Input")
TARGET_PATH = os.path.join(OUTPUT_PATH, "Target")

# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 3
NUM_LEVELS = 1
# initialize learning rate, number of epochs to train for, and the
# batch size
#INIT_LR = #0.00005
INIT_LR = 0.001#0.0005
NUM_EPOCHS = 3
BATCH_SIZE = 24
#BATCH_SIZE = 128
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 64
# define threshold to filter weak predictions
THRESHOLD = 0.5

