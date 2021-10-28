import torch
import random
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

random.seed(0)

from loss_functions import LossFunction
from model import BertBaselineClassifier

argp = argparse.ArgumentParser()
argp.add_argument('--train',
                  help="Whether to train or evaluate a model", default=False)
argp.add_argument('--eval',
                  help="Whether to train or evaluate a model", default=False)
argp.add_argument('--path_to_saved_model',
                  help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--path_to_test_data',
                  help="Path to save the model after pretraining/finetuning", default=None)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = BertBaselineClassifier(weights_name='bert-base-cased',
                               n_classes_=2,  # binary classification
                               batch_size=8,  # Small batches to avoid memory overload.
                               max_iter=4)  # We'll search based on 1 iteration for efficiency.)

# Initialize data
data = pd.read_csv(
    '/Users/blakenorwick/Stanford_XCS224U/project/data/AgreementDataset/agreement_dataset_valid_tweets.tsv', sep='\t')
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if args.train:
    loss_function = LossFunction()
    model.fit(loss_function, X_train, y_train)
    model.to_pickle('/Users/blakenorwick/Stanford_XCS224U/project/baseline_model_test.pkl')

if args.eval:
    model = model.from_pickle('/Users/blakenorwick/Stanford_XCS224U/project/baseline_model_test.pkl')
    predictions = model.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, predictions, digits=3, output_dict=True)).transpose()
    report.to_csv('/Users/blakenorwick/Stanford_XCS224U/project/report.tsv', sep='\t')

    # use F1 as score metric in baseline, so this is redundant
    # scores = utils.safe_macro_f1(y_test[:50], predictions)
    # mean_score = np.mean(scores)
    # print("Mean of macro-F1 scores: {0:.03f}".format(mean_score))
