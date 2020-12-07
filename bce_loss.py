import argparse
import os
import time

import pandas as pd
import numpy as np
# import sys
# sys.path.append('../')

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import ramps
from DatasetDcase2019Task4_weak2strong_NN_ex import DatasetDcase2019Task4_weak2strong_NN_ex
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel_weak2strong_NN_ex import test_model
from evaluation_measures_weak2strong_NN_ex import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config_weak2strong_NN_ex as cfg
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, get_transforms_AANPT, get_transforms_nopad, AverageMeterSet
from utils.Logger import LOG, TIME

from tensorboardX import SummaryWriter

loss = nn.BCELoss()
a = torch.tensor(0.3)
b = torch.tensor(1.)
print(loss(a, b))
print(torch.log(a))
