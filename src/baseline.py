import torch
import numpy as np
from torch import nn
from util import mean_pooling, convert_numeral_to_eight_levels
from model_base import LevelEstimaterBase


class BaselineClassification(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, attach_wlv, num_labels,
                 word_num_labels, alpha,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, False, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.dropout = nn.Dropout(0.1)

        if self.problem_type == 'baseline_reg':
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, 1)
            self.loss_fct = nn.MSELoss()
        else:
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, self.num_labels)
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # in lightning, forward defines the prediction/inference actions
        outputs, _ = self.encode(inputs)
        outputs = mean_pooling(outputs, attention_mask=inputs['attention_mask'])
        logits = self.slv_classifier(self.dropout(outputs))

        loss = None
        if 'labels' in inputs:
            if self.problem_type == "baseline_reg":
                labels = inputs['labels']
                cls_loss = self.loss_fct(logits.squeeze(), labels.squeeze())
            else:
                labels = inputs['labels'].detach().clone()
                cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))
            logs = {"loss": loss}

        if self.problem_type == "baseline_reg":
            predictions = convert_numeral_to_eight_levels(logits.cpu().detach().clone().numpy())
        else:
            predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True).cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions

    def step(self, batch):
        loss, predictions, logs = self.forward(batch)
        return loss, logs

    def _shared_eval_step(self, batch):
        loss, predictions, logs = self.forward(batch)

        gold_labels = batch['labels'].cpu().detach().clone().numpy()
        golds_predictions = {'gold_labels': gold_labels,
                             'pred_labels': predictions}

        return logs, golds_predictions

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return golds_predictions

    def validation_epoch_end(self, outputs):
        logs = self.evaluation(outputs)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

    def test_step(self, batch, batch_idx):
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return golds_predictions

    def test_epoch_end(self, outputs):
        logs = self.evaluation(outputs, test=True)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
