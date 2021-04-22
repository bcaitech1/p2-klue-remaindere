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

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    print(f"lables : {labels}")
    preds = pred.predictions.argmax(-1)
    print(f"preds : {preds}")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def train(args):
  #seed holding 
  seed_everything(seed = 42)
  #train device define
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

  # load dataset
  total_dataset = load_data("/opt/ml/input/data/train/data.tsv")
  total_label = total_dataset['label'].values
  
  #for weighted loss
  #   df_label = pd.Series(total_label)
  #   label_sorted = df_label.value_counts().sort_index()
  #   n_label = torch.Tensor(label_sorted.values)
  #   gamma = 2
  #   normed_weights = [1 - (gamma*x/sum(n_label)) for x in n_label]
  #   normed_weights = torch.FloatTensor(normed_weights).to(device)
  #   global loss_weight 
  #   #print(normed_weights)
  #   loss_weight = normed_weights
  
  # tokenizing dataset
  tokenized_data = tokenized_dataset(total_dataset, tokenizer)
  # make dataset for pytorch.
  RE_dataset = RE_Dataset(tokenized_data, total_label)
        
  #kfold
  kfold = KFold(n_splits = 5,random_state = 42, shuffle = True)
  folds = []
  #set idx from labels
  for fold_index, (train_idx, valid_idx) in tqdm(enumerate(kfold.split(range(len(total_label))))) :
    folds.append({'train' : train_idx, 'valid' : valid_idx})

  print()
  print(f'total fold: {len(folds)}')
  fold_count = 0
  for fold in tqdm(folds) :
    fold_count += 1
    if fold_count < 4 :
        continue
    print(f'current fold: {fold_count}, fold val: {fold}')
    #print("train_idx check starts...")
    #for t in train_idx :
    #     print(t)
    #print("valid_idx check starts...")
    #for v in valid_idx :
    #     print(v)
    #train & valid subset define
    train_subset = Subset(dataset=RE_dataset, indices=train_idx)
    valid_subset = Subset(dataset=RE_dataset, indices=valid_idx)
    
    # setting model hyperparameter
    XLMRoberta_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    XLMRoberta_config.num_labels = 42
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=XLMRoberta_config)
    model.parameters
    
    #set optimizer & scheduler
    #optimizer = madgrad.MADGRAD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = args.weight_decay, eps = 1e-06)
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.8)
    
    #load model to device
    model.to(device)
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
    training_args = TrainingArguments(
      output_dir=os.path.join(args.output_dir,str(fold_count)+"th_fold"),          # output directory
      save_total_limit=args.save_total_limit,              # number of total save model.
      save_steps=args.save_steps,                 # model saving step (if u want to save model at [savestep*n] step)
      #save_strategy='epoch',
      evaluation_strategy='steps',
      eval_steps=100,
      num_train_epochs=args.num_train_epochs,              # total number of training epochs
      learning_rate=args.learning_rate,               # learning_rate
      per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
      per_device_eval_batch_size=128,
      #warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
      weight_decay=args.weight_decay,               # strength of weight decay
      logging_dir=args.logging_dir,            # directory for storing logs
      logging_steps=args.logging_steps,              # log saving step.
      label_smoothing_factor = 0.5,
    )
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      #optimizers=optimizer, #custom opt, scheduler
      train_dataset=train_subset,         # training dataset
      eval_dataset=valid_subset,          # eval dataset
      compute_metrics=compute_metrics,     # compute_metrics
    )
    # train model
    trainer.train()

def main(args):
  train(args)

def seed_everything(seed) :
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Train Hyperparams")
  parser.add_argument('--output_dir', required=True, type=str, default="/opt/ml/results/")
  parser.add_argument('--save_total_limit', required=False, type=int, default=2)
  parser.add_argument('--save_steps', required=False, type=int, default=100)
  parser.add_argument('--num_train_epochs', required=False, type=int, default=5)
  parser.add_argument('--learning_rate', required=False, type=float, default=4e-5)
  parser.add_argument('--per_device_train_batch_size', required=False, type=int, default=64)
  #parser.add_argument('--lr_decay_step', required=False, type=int, default=120)
  #parser.add_argument('--warmup_steps', required=False, type=int, default=450)
  parser.add_argument('--weight_decay', required=False, type=float, default=0.005)
  parser.add_argument('--logging_dir', required=False, type=str, default="/opt/ml/logs/")
  parser.add_argument('--logging_steps', required=False, type=int, default=25)
  args = parser.parse_args()
  print(args)
  main(args)
