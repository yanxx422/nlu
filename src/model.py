import utils
import torch
import torch.nn as nn
import numpy as np
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

    def __init__(self, weights_name,
                 use_empirical_data=False,
                 gradient_accumulation_steps=8,
                 eta=0.00001,
                 loss=torch.nn.CrossEntropyLoss(),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_name = weights_name
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.eta = eta
        self.use_empirical_data = use_empirical_data
        self.loss = loss
        self.tokenizer = BertTokenizer.from_pretrained(weights_name)

    def build_graph(self):
        # computation graph
        # sets model in TorchShallowNeuralClassifier fit
        return BertBaselineModel(self.weights_name, self.n_classes_)

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
            if self.use_empirical_data:
                self.n_classes_ = len(y[-1])
                self.classes_ = [item for item in range(self.n_classes_)]
                dataset = torch.utils.data.TensorDataset(indices, mask, torch.tensor(y))
            else:
                self.classes_ = sorted(set(y))
                self.n_classes_ = len(self.classes_)
                class2index = dict(zip(self.classes_, range(self.n_classes_)))
                y = [class2index[label] for label in y]
                y = torch.tensor(y)
                dataset = torch.utils.data.TensorDataset(indices, mask, torch.tensor(y))
        return dataset

    def fit(self, *args):
        """
        Generic optimization method.

        Parameters
        ----------
        *args: list of objects
            We assume that the final element of args give the labels
            and all the preceding elements give the system inputs.
            For regular supervised learning, this is like (X, y), but
            we allow for models that might use multiple data structures
            for their inputs.

        Attributes
        ----------
        model: nn.Module or subclass thereof
            Set by `build_graph`. If `warm_start=True`, then this is
            initialized only by the first call to `fit`.

        optimizer: torch.optimizer.Optimizer
            Set by `build_optimizer`. If `warm_start=True`, then this is
            initialized only by the first call to `fit`.

        errors: list of float
            List of errors. If `warm_start=True`, then this is
            initialized only by the first call to `fit`. Thus, where
            `max_iter=5`, if we call `fit` twice with `warm_start=True`,
            then `errors` will end up with 10 floats in it.

        validation_scores: list
            List of scores. This is filled only if `early_stopping=True`.
            If `warm_start=True`, then this is initialized only by the
            first call to `fit`. Thus, where `max_iter=5`, if we call
            `fit` twice with `warm_start=True`, then `validation_scores`
            will end up with 10 floats in it.

        no_improvement_count: int
            Used to control early stopping and convergence. These values
            are controlled by `_update_no_improvement_count_early_stopping`
            or `_update_no_improvement_count_errors`.  If `warm_start=True`,
            then this is initialized only by the first call to `fit`. Thus,
            in that situation, the values could accumulate across calls to
            `fit`.

        best_error: float
           Used to control convergence. Smaller is assumed to be better.
           If `warm_start=True`, then this is initialized only by the first
           call to `fit`. It will be reset by
           `_update_no_improvement_count_errors` depending on how the
           optimization is proceeding.

        best_score: float
           Used to control early stopping. If `warm_start=True`, then this
           is initialized only by the first call to `fit`. It will be reset
           by `_update_no_improvement_count_early_stopping` depending on how
           the optimization is proceeding. Important: we currently assume
           that larger scores are better. As a result, we will not get the
           correct results for, e.g., a scoring function based in
           `mean_squared_error`. See `self.score` for additional details.

        best_parameters: dict
            This is a PyTorch state dict. It is used if and only if
            `early_stopping=True`. In that case, it is updated whenever
            `best_score` is improved numerically. If the early stopping
            criteria are met, then `self.model` is reset to contain these
            parameters before `fit` exits.

        Returns
        -------
        self

        """
        if self.early_stopping:
            args, dev = self._build_validation_split(
                *args, validation_fraction=self.validation_fraction)

        # Dataset:
        dataset = self.build_dataset(*args)
        dataloader = self._build_dataloader(dataset, shuffle=True)

        # Graph:
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.build_graph()
            # This device move has to happen before the optimizer is built:
            # https://pytorch.org/docs/master/optim.html#constructing-it
            self.model.to(self.device)
            self.optimizer = self.build_optimizer()
            self.errors = []
            self.validation_scores = []
            self.no_improvement_count = 0
            self.best_error = np.inf
            self.best_score = -np.inf
            self.best_parameters = None

        # Make sure the model is where we want it:
        self.model.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        for iteration in range(1, self.max_iter+1):

            epoch_error = 0.0

            for batch_num, batch in enumerate(dataloader, start=1):

                batch = [x.to(self.device, non_blocking=True) for x in batch]

                X_batch = batch[:-1]
                y_batch = batch[-1]

                batch_preds = self.model(*X_batch)

                err = self.loss(batch_preds, y_batch)

                if self.gradient_accumulation_steps > 1 and \
                  self.loss.reduction == "mean": err /= self.gradient_accumulation_steps

                err.backward()

                epoch_error += err.item()

                if batch_num % self.gradient_accumulation_steps == 0 or batch_num == len(dataloader):
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Stopping criteria:

            if self.early_stopping:
                self._update_no_improvement_count_early_stopping(*dev)
                if self.no_improvement_count > self.n_iter_no_change:
                    utils.progress_bar(
                        "Stopping after epoch {}. Validation score did "
                        "not improve by tol={} for more than {} epochs. "
                        "Final error is {}".format(iteration, self.tol,
                            self.n_iter_no_change, epoch_error),
                        verbose=self.display_progress)
                    break

            else:
                self._update_no_improvement_count_errors(epoch_error)
                if self.no_improvement_count > self.n_iter_no_change:
                    utils.progress_bar(
                        "Stopping after epoch {}. Training loss did "
                        "not improve more than tol={}. Final error "
                        "is {}.".format(iteration, self.tol, epoch_error),
                        verbose=self.display_progress)
                    break

            utils.progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error),
                verbose=self.display_progress)

        if self.early_stopping:
            self.model.load_state_dict(self.best_parameters)

        return self

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

