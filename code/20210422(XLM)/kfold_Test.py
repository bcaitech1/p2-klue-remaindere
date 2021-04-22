import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, Trainer, TrainingArguments
from load_data import *
import numpy as np
import random
import argparse
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaTokenizer
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import madgrad

def train():
  kfold = KFold(n_splits = 5)
  folds = []
  #set idx from labels
  for fold_index, (train_idx, valid_idx) in tqdm(enumerate(kfold.split(range(9000)))) :
    folds.append({'train' : train_idx, 'valid' : valid_idx})

  print()
  print(f'total fold: {len(folds)}')
  fold_count = 0
  for fold in folds :
    print(fold)
def main():
  train()

if __name__ == '__main__':
  main()
