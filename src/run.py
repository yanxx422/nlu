import torch
import random
import argparse
import pandas as pd
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loss_functions import LabelSmoothingLoss

random.seed(0)

from model import BertBaselineClassifier

argp = argparse.ArgumentParser()
argp.add_argument('--train',
                  help="Whether to train or evaluate a model", default=False)
argp.add_argument('--eval',
                  help="Whether to train or evaluate a model", default=False)
argp.add_argument('--use_empirical_labels',
                  help="use empirical labels instead of one-hot encoding", default=False)
argp.add_argument('--path_to_saved_model',
                  help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--path_to_test_data',
                  help="Path to save the model after pretraining/finetuning", default=None)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = BertBaselineClassifier(weights_name='bert-base-cased',
                               n_classes_=2,
                               batch_size=64)

# Initialize data
data = pd.read_csv(
    '/Users/blakenorwick/Stanford_XCS224U/project/data/AgreementDataset/agreement_dataset_valid_tweets.tsv', sep='\t')
X = data['text']
y = data['label']
confidence = data['agreement_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if args.train:
    smoothed_labels = utils.smooth_labels(y, confidence)
    model.fit(X_train, y_train)
    model.to_pickle('/Users/blakenorwick/Stanford_XCS224U/project/baseline_model_test.pkl')

if args.eval:
    model = model.from_pickle('/Users/blakenorwick/Stanford_XCS224U/project/baseline_model_test.pkl')
    predictions = model.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, predictions, digits=3, output_dict=True)).transpose()
    report.to_csv('/Users/blakenorwick/Stanford_XCS224U/project/report.tsv', sep='\t')

if args.use_empirical_labels:
    smoothed_labels = utils.smooth_labels(y, confidence)
    model.fit(X_train, smoothed_labels)
    model.fit(X_train, smoothed_labels)
    model.to_pickle('/Users/blakenorwick/Stanford_XCS224U/project/baseline_model_test.pkl')
