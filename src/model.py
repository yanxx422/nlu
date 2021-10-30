import utils
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier


class BertBaselineModel(nn.Module):
    def __init__(self, weights_name, n_classes_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = BertModel.from_pretrained(weights_name)
        self.bert.train()
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.linear_layer = nn.Linear(self.hidden_dim, n_classes_)

    def forward(self, indices, mask):
        bert_representations = self.bert(input_ids=indices, attention_mask=mask)
        return self.linear_layer(bert_representations.pooler_output)

class BertBaselineClassifier(TorchShallowNeuralClassifier):

    def __init__(self, weights_name, n_classes_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_name = weights_name
        self.n_classes_ = n_classes_
        self.tokenizer = BertTokenizer.from_pretrained(weights_name)

    def build_graph(self):
        # computation graph
        # sets model in TorchShallowNeuralClassifier fit
        return BertBaselineModel(self.weights_name, self.n_classes_)

# TODO: modify to use empirical distribution as labels as in Ex Machina: Personal Attacks Seen at Scale?
    def build_dataset(self, X, y=None):
        data = self.tokenizer.batch_encode_plus(
            X,
            max_length=None,
            add_special_tokens=True,
            padding='longest',
            return_attention_mask=True)
        indices = torch.tensor(data['input_ids'])
        mask = torch.tensor(data['attention_mask'])
        if y is None:
            dataset = torch.utils.data.TensorDataset(indices, mask)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            print(y.size())
            print(indices.size())
            dataset = torch.utils.data.TensorDataset(indices, mask, y)
        return dataset


    # TODO: Implement score() in case we adopt metrics proposed in http://www.kayur.org/papers/chi2021.pdf
    def score(self, X, y, device=None):
        """
        Uses macro-F1 as the score function.

        This function can be used to evaluate models, but its primary
        use is in cross-validation and hyperparameter tuning.

        Parameters
        ----------
        X: np.array, shape `(n_examples, n_features)`

        y: iterable, shape `len(n_examples)`
            These can be the raw labels. They will converted internally
            as needed. See `build_dataset`.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        float

        """
        preds = self.predict(X, device=device)
        return utils.safe_macro_f1(y, preds)

