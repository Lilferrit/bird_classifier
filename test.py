from PIL import Image
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch import nn

import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

print("Holy shit no way lmao")