import torch

""" Dataset """
RESIZE = (64, 64) # Smaller images to train faster
DATA = "cats"
DATA_FORMAT = "*.jpg"

""" Diffusion """ # I used the default values from the paper
T = 200
BETA_START = 0.0001
BETA_END = 0.02

""" Training """
DTYPE = torch.float32
BATCH_SIZE = 250
EMBEDDING_DIM = 32
LR = 0.001
EPOCHS = 500
SAVE_STEP = 50 # Needs to be smaller than the number of epochs
OUT_WEIGHTS = "output_weights"
LOAD_WEIGHTS = False
WEIGHT_TO_LOAD = "model_epoch_15.pth"

""" Sampling """
OUT_DIR = "output_samples"
N_IM = 10
