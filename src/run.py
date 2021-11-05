import torch
import random
import argparse
import pandas as pd
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loss_functions import LabelSmoothing, FocalLoss

random.seed(0)

from model import BertBaselineClassifier

argp = argparse.ArgumentParser()
argp.add_argument('--baseline',
                  help="Whether to train or evaluate a model", default=False)
argp.add_argument('--eval',
                  help="Whether to train or evaluate a model", default=False)
argp.add_argument('--use_empirical_labels',
                  help="use empirical labels instead of one-hot encoding", default=False)
argp.add_argument('--path_to_save_model',
                  help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--path_to_report',
                  help="Path to model report", default=None)
argp.add_argument('--path_to_data',
                  help="Path to data", default=None)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Initialize data
data = pd.read_csv(args.path_to_data, sep='\t')
X = data['text']
y = data['label']
confidence = data['agreement_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if args.baseline:
    model = BertBaselineClassifier(weights_name='bert-base-cased',
                                   loss=FocalLoss(),
                                   batch_size=32,
                                   max_iter=2)
    model.fit(X_train, y_train)
    model.to_pickle(args.path_to_save_model)

if args.use_empirical_labels:
    model = BertBaselineClassifier(weights_name='bert-base-cased',
                                   loss=torch.nn.BCEWithLogitsLoss(reduction='mean'),
                                   use_empirical_data=True,
                                   batch_size=32,
                                   max_iter=2)
    smoothed_labels = utils.smooth_labels(y_train, confidence)
    model.fit(X_train, smoothed_labels)
    model.to_pickle(args.path_to_save_model)

if args.eval:
    model = model.from_pickle(args.path_to_save_model)
    predictions = model.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, predictions, digits=3, output_dict=True)).transpose()
    report.to_csv(args.path_to_report, sep='\t')
