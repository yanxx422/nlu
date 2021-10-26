import torch
import random
import argparse
import pandas as pd

from tqdm import tqdm

random.seed(0)

from dataset import AgreementData
from loss_functions import LossFunction
from model import BertBaselineClassifier

argp = argparse.ArgumentParser()
argp.add_argument('--function',
                  help="Whether to train or evaluate a model",
                  choices=["train", "evaluate"])
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
                               max_iter=1)  # We'll search based on 1 iteration for efficiency.)

if args.function == 'train':
    data = pd.read_csv('/Users/blakenorwick/Stanford_XCS224U/project/data/AgreementDataset/agreement_dataset_valid_tweets.tsv', sep='\t')
    loss_function = LossFunction()
    model.fit(loss_function, data['text'], data['label'])

if args.function == 'evaluate':

    model.load_state_dict(torch.load(args.path_to_saved_model))
    model = model.to(device)
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.path_to_test_data, encoding='utf-8')):
            # TODO
            pass
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
