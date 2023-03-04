import torch

""" Dataset """
RESIZE = (64, 64) # Smaller images to train faster
DATA = "cats"
DATA_FORMAT = "*.jpg"

""" Diffusion """ # I used the default values from the paper
T = 1000
BETA_START = 0.0001
BETA_END = 0.02

""" Training """
DTYPE = torch.float32
BATCH_SIZE = 200
EMBEDDING_DIM = 32
L_RATE = 0.001
EPOCHS = 500
SAVE_STEP = 10 # Needs to be smaller than the number of epochs
OUT_WEIGHTS = "output_weights"
LOAD_WEIGHTS = False
WEIGHT_TO_LOAD = "model_epoch_15.pth"

""" Sampling """
OUT_DIR = "output_samples"
N_IM = 10
