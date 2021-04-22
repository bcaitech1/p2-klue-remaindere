import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *
import numpy as np
import random
from transformers import ElectraForSequenceClassification, ElectraConfig, ElectraTokenizer
import argparse

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train(args):
  # load model and tokenizer
  MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
  tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  train_label = train_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  electra_config = ElectraConfig.from_pretrained(MODEL_NAME)
  electra_config.num_labels = 42
  model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, config=electra_config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
  training_args = TrainingArguments(
    output_dir=args.output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.num_train_epochs,              # total number of training epochs
    learning_rate=args.learning_rate,               # learning_rate
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir=args.logging_dir,            # directory for storing logs
    logging_steps=args.logging_steps,              # log saving step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
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
  seed_everything(seed = 42)
  parser = argparse.ArgumentParser(description="Train Hyperparams")
  parser.add_argument('--output_dir', required=True, type=str, default="/opt/ml/results/")
  parser.add_argument('--save_total_limit', required=False, type=int, default=20)
  parser.add_argument('--save_steps', required=False, type=int, default=500)
  parser.add_argument('--num_train_epochs', required=True, type=int, default=2)
  parser.add_argument('--learning_rate', required=True, type=float, default=5e-5)
  parser.add_argument('--per_device_train_batch_size', required=False, type=int, default=16)
  parser.add_argument('--warmup_steps', required=False, type=int, default=500)
  parser.add_argument('--weight_decay', required=False, type=float, default=0.001)
  parser.add_argument('--logging_dir', required=False, type=str, default="/opt/ml/logs/")
  parser.add_argument('--logging_steps', required=False, type=int, default=100)
  args = parser.parse_args()
  print(args)
  main(args)
