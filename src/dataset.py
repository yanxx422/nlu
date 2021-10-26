import argparse
import torch
from transformers import BertTokenizer

# TODO: in case we wanna move the train functionality outside of model.fit()
class AgreementData():
    def __init__(self, data):
        self.encodings, self.attn_mask, self.labels = self.tokenize(data)

    def tokenize(self, data):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        tokens = tokenizer.batch_encode_plus(list(data["text"]), max_length=512, padding='max_length', truncation=True)
        text_seq = torch.tensor(tokens['input_ids'])
        text_mask = torch.tensor(tokens['attention_mask'])
        labels = data['label']
        return text_seq, text_mask, labels

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encodings[idx]).squeeze()
        target_ids = torch.tensor(self.labels[idx]).squeeze()
        attention_mask = torch.tensor(self.attn_mask[idx]).squeeze()
        return {"input_ids": input_ids, "labels": target_ids, "attention_mask": attention_mask}

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type',
                      help="Type of dataset to sample from.",
                      choices=["agreementdata"])
    args = argp.parse_args()

    if args.dataset_type == 'agreementdata':
        agreement_dataset = AgreementData('/Users/blakenorwick/Stanford_XCS224U/project/data/AgreementDataset/agreement_dataset_valid_tweets.tsv')
        for i, row in agreement_dataset[:10].iterrows():
            print(row['text'])