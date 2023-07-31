import torch, random, itertools, tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from util import mean_pooling, read_corpus, CEFRDataset, convert_numeral_to_eight_levels
from model_base import LevelEstimaterBase
from losses import *

class LevelEstimaterClassification(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels, alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.normalize_cls = args.normalize_cls

        if self.problem_type == "regression":
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, 1)
            self.loss_fct = nn.MSELoss()
            self.score_bins 
        else:
            if args.use_layernorm:
                self.slv_classifier = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                                        nn.LayerNorm(self.lm.config.hidden_size),
                                                        nn.Linear(self.lm.config.hidden_size, self.CEFR_lvs, bias=(not self.normalize_cls)))
            else:
                self.slv_classifier = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                                        nn.Linear(self.lm.config.hidden_size, self.CEFR_lvs, bias=(not self.normalize_cls)))
        
            if self.with_loss_weight:
                train_sentlv_weights = self.precompute_loss_weights()
                self.loss_fct = nn.CrossEntropyLoss(weight=train_sentlv_weights)
            else:
                self.loss_fct = nn.CrossEntropyLoss()

            if args.loss_type != "cross_entropy":
                assert self.with_loss_weight == False
                self.loss_fct = eval(args.loss_type)(args.CEFR_lvs)

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs_mean = mean_pooling(outputs, attention_mask=batch['attention_mask'])
        
        if self.normalize_cls:
            for W in self.slv_classifier.parameters():
                W = F.normalize(W, dim=1)

        logits = self.slv_classifier(outputs_mean)

        if self.problem_type == "regression":
            predictions = convert_numeral_to_eight_levels(logits.detach().clone().cpu().numpy())
        else:
            predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'labels' in batch:
            if self.problem_type == "regression":
                labels = batch['labels']
                cls_loss = self.loss_fct(logits.squeeze(), labels.squeeze())
            else:
                labels = batch['labels'].detach().clone()
                cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": cls_loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions

class LevelEstimaterCORN(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels, alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.normalize_cls = args.normalize_cls

        if args.use_layernorm:
            self.slv_classifier = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                                    nn.LayerNorm(self.lm.config.hidden_size),
                                                    nn.Linear(self.lm.config.hidden_size, self.CEFR_lvs-1))
        else:
            self.slv_classifier = nn.Sequential(nn.Dropout(p=args.dropout_rate),
                                                    nn.Linear(self.lm.config.hidden_size, self.CEFR_lvs-1))
        
        if self.with_loss_weight:
            train_sentlv_weights = self.precompute_loss_weights()
            self.loss_fct = CORNLoss(args.CEFR_lvs, weight=train_sentlv_weights)
        else:
            self.loss_fct = CORNLoss(args.CEFR_lvs)
    
    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs_mean = mean_pooling(outputs, attention_mask=batch['attention_mask'])

        logits = self.slv_classifier(outputs_mean)
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        predict_levels = probas > 0.5
        predictions = torch.sum(predict_levels, dim=1).unsqueeze(-1)

        loss = None
        if 'labels' in batch:
            labels = batch['labels'].detach().clone()
            cls_loss = self.loss_fct(logits, labels.view(-1))

            loss = cls_loss
            logs = {"loss": cls_loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions


class LevelEstimaterContrastive(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels,
                 num_prototypes,
                 alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(args.dropout_rate)

        self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size)
        self.init_prototypes = args.init_prototypes
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        # nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        assert args.use_layernorm == False            
        if args.loss_type != "cross_entropy":
            if self.with_loss_weight:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs, loss_weights=loss_weights) 
            else:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs) 

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs = self.dropout(outputs)
        outputs = mean_pooling(outputs, attention_mask=batch['attention_mask'])
        #outputs = self.dropout(mean_pooling(outputs, attention_mask=batch['attention_mask']))

        # positive: compute cosine similarity
        outputs = torch.nn.functional.normalize(outputs)
        positive_prototypes = torch.nn.functional.normalize(self.prototype.weight)
        logits = torch.mm(outputs, positive_prototypes.T)
        logits = logits.reshape((-1, self.num_prototypes, self.CEFR_lvs))
        logits = logits.mean(dim=1)

        # prediction
        predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'labels' in batch:
            labels = batch['labels'].detach().clone()
            # cross-entropy loss
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions

    def on_train_start(self) -> None:
        # Init with BERT embeddings
        if self.init_prototypes == "pretrained":
            print("init prototypes from pretrained")
        else:
            print("init prototypes", self.init_prototypes)
            return

        epcilon = 1.0e-6
        labels = []
        prototype_initials = torch.full((self.CEFR_lvs, self.lm.config.hidden_size), fill_value=epcilon).to(self.device)

        self.lm.eval()
        for batch in tqdm.tqdm(self.train_dataloader(), leave=False, desc='init prototypes'):
            labels += batch['labels'].squeeze(-1).detach().clone().numpy().tolist()
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                outputs_mean = mean_pooling(outputs.hidden_states[self.lm_layer],
                                            attention_mask=batch['attention_mask'])
            for lv in range(self.CEFR_lvs):
                prototype_initials[lv] += outputs_mean[
                    batch['labels'].squeeze(-1) == lv].sum(0)
        if not self.with_ib:
            self.lm.train()

        labels = torch.tensor(labels)
        for lv in range(self.CEFR_lvs):
            denom = torch.count_nonzero(labels == lv) + epcilon
            prototype_initials[lv] = prototype_initials[lv] / denom

        var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
        # prototype_initials = torch.repeat_interleave(prototype_initials, self.num_prototypes, dim=0)
        prototype_initials = prototype_initials.repeat(self.num_prototypes, 1)
        noise = (var ** 0.5) * torch.randn(prototype_initials.size()).to(self.device)
        prototype_initials = prototype_initials + noise  # Add Gaussian noize
        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # # Init with Xavier
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization

class LevelEstimaterContrastiveSED(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels,
                 num_prototypes,
                 alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(args.dropout_rate)

        self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size)
        self.init_prototypes = args.init_prototypes
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        # nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        assert args.use_layernorm == False            
        if args.loss_type != "cross_entropy":
            if self.with_loss_weight:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs, loss_weights=loss_weights) 
            else:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs) 
    
    def negative_sed(self, a, b):
        ''' negative square euclidean distance
        - input
            a: batch x D
            b: (num_label * num_proto) x D
        - output
            logits: batch x num_label
        '''

        # calculate centroid of prototypes
        b = b.reshape(self.num_labels, self.num_prototypes, -1)
        b = b.mean(dim=1)

        n = a.shape[0]
        m = b.shape[0]
        if a.size(1) != b.size(1):
            raise Exception
     
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)

        return logits

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs = self.dropout(outputs)
        outputs = mean_pooling(outputs, attention_mask=batch['attention_mask'])
        #outputs = self.dropout(mean_pooling(outputs, attention_mask=batch['attention_mask']))

        # positive: compute cosine similarity
        logits = self.negative_sed(outputs, self.prototype.weight)

        # prediction
        predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'labels' in batch:
            labels = batch['labels'].detach().clone()
            # cross-entropy loss
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions
    
    def on_train_start(self) -> None:
        # Init with BERT embeddings
        if self.init_prototypes == "pretrained":
            print("init prototypes from pretrained")
        else:
            print("init prototypes", self.init_prototypes)
            return
            
        epcilon = 1.0e-6
        labels = []
        prototype_initials = torch.full((self.CEFR_lvs, self.lm.config.hidden_size), fill_value=epcilon).to(self.device)

        self.lm.eval()
        for batch in tqdm.tqdm(self.train_dataloader(), leave=False, desc='init prototypes'):
            labels += batch['labels'].squeeze(-1).detach().clone().numpy().tolist()
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                outputs_mean = mean_pooling(outputs.hidden_states[self.lm_layer],
                                            attention_mask=batch['attention_mask'])
            for lv in range(self.CEFR_lvs):
                prototype_initials[lv] += outputs_mean[
                    batch['labels'].squeeze(-1) == lv].sum(0)
        if not self.with_ib:
            self.lm.train()

        labels = torch.tensor(labels)
        for lv in range(self.CEFR_lvs):
            denom = torch.count_nonzero(labels == lv) + epcilon
            prototype_initials[lv] = prototype_initials[lv] / denom

        var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
        # prototype_initials = torch.repeat_interleave(prototype_initials, self.num_prototypes, dim=0)
        prototype_initials = prototype_initials.repeat(self.num_prototypes, 1)
        noise = (var ** 0.5) * torch.randn(prototype_initials.size()).to(self.device)
        prototype_initials = prototype_initials + noise  # Add Gaussian noize
        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # # Init with Xavier
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization

class LevelEstimaterContrastiveDot(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels,
                 num_prototypes,
                 alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(args.dropout_rate)

        self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size)
        self.init_prototypes = args.init_prototypes
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        # nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        assert args.use_layernorm == False            
        if args.loss_type != "cross_entropy":
            if self.with_loss_weight:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs, loss_weights=loss_weights) 
            else:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs) 

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs = self.dropout(outputs)
        outputs = mean_pooling(outputs, attention_mask=batch['attention_mask'])
        #outputs = self.dropout(mean_pooling(outputs, attention_mask=batch['attention_mask']))

        # positive: compute cosine similarity
        outputs = outputs #torch.nn.functional.normalize(outputs)
        positive_prototypes = self.prototype.weight #torch.nn.functional.normalize(self.prototype.weight)
        logits = torch.mm(outputs, positive_prototypes.T)
        logits = logits.reshape((-1, self.num_prototypes, self.CEFR_lvs))
        logits = logits.mean(dim=1)

        # prediction
        predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'labels' in batch:
            labels = batch['labels'].detach().clone()
            # cross-entropy loss
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions
    
    def on_train_start(self) -> None:
        # Init with BERT embeddings
        if self.init_prototypes == "pretrained":
            print("init prototypes from pretrained")
        else:
            print("init prototypes", self.init_prototypes)
            return

        epcilon = 1.0e-6
        labels = []
        prototype_initials = torch.full((self.CEFR_lvs, self.lm.config.hidden_size), fill_value=epcilon).to(self.device)

        self.lm.eval()
        for batch in tqdm.tqdm(self.train_dataloader(), leave=False, desc='init prototypes'):
            labels += batch['labels'].squeeze(-1).detach().clone().numpy().tolist()
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                outputs_mean = mean_pooling(outputs.hidden_states[self.lm_layer],
                                            attention_mask=batch['attention_mask'])
            for lv in range(self.CEFR_lvs):
                prototype_initials[lv] += outputs_mean[
                    batch['labels'].squeeze(-1) == lv].sum(0)
        if not self.with_ib:
            self.lm.train()

        labels = torch.tensor(labels)
        for lv in range(self.CEFR_lvs):
            denom = torch.count_nonzero(labels == lv) + epcilon
            prototype_initials[lv] = prototype_initials[lv] / denom

        var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
        # prototype_initials = torch.repeat_interleave(prototype_initials, self.num_prototypes, dim=0)
        prototype_initials = prototype_initials.repeat(self.num_prototypes, 1)
        noise = (var ** 0.5) * torch.randn(prototype_initials.size()).to(self.device)
        prototype_initials = prototype_initials + noise  # Add Gaussian noize
        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # # Init with Xavier
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
    
class LevelEstimaterSContrastive(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels,
                 num_prototypes,
                 alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(args.dropout_rate)

        #self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size, max_norm=1)
        self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size)
        self.init_prototypes = args.init_prototypes
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        # nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal
        self.slv_w = nn.Parameter(torch.tensor(10.0))
        self.slv_b = nn.Parameter(torch.tensor(-5.0))

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        assert args.use_layernorm == False            
        if args.loss_type != "cross_entropy":
            if self.with_loss_weight:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs, loss_weights=loss_weights) 
            else:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs) 

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs = self.dropout(outputs)
        outputs = mean_pooling(outputs, attention_mask=batch['attention_mask']) 
        
        # positive: compute cosine similarity
        outputs = torch.nn.functional.normalize(outputs)
        positive_prototypes = torch.nn.functional.normalize(self.prototype.weight)
        logits = torch.mm(outputs, positive_prototypes.T)
        logits = logits.reshape((-1, self.num_prototypes, self.CEFR_lvs))
        logits = logits.mean(dim=1)
        logits = self.slv_w * logits + self.slv_b

        # prediction
        predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'labels' in batch:
            labels = batch['labels'].detach().clone()
            # cross-entropy loss
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions
    
    def on_train_start(self) -> None:
        # Init with BERT embeddings
        if self.init_prototypes == "pretrained":
            print("init prototypes from pretrained")
        else:
            print("init prototypes", self.init_prototypes)
            return

        epcilon = 1.0e-6
        labels = []
        prototype_initials = torch.full((self.CEFR_lvs, self.lm.config.hidden_size), fill_value=epcilon).to(self.device)

        self.lm.eval()
        for batch in tqdm.tqdm(self.train_dataloader(), leave=False, desc='init prototypes'):
            labels += batch['labels'].squeeze(-1).detach().clone().numpy().tolist()
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                outputs_mean = mean_pooling(outputs.hidden_states[self.lm_layer],
                                            attention_mask=batch['attention_mask'])
            for lv in range(self.CEFR_lvs):
                prototype_initials[lv] += outputs_mean[
                    batch['labels'].squeeze(-1) == lv].sum(0)
        if not self.with_ib:
            self.lm.train()

        labels = torch.tensor(labels)
        for lv in range(self.CEFR_lvs):
            denom = torch.count_nonzero(labels == lv) + epcilon
            prototype_initials[lv] = prototype_initials[lv] / denom

        var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
        # prototype_initials = torch.repeat_interleave(prototype_initials, self.num_prototypes, dim=0)
        prototype_initials = prototype_initials.repeat(self.num_prototypes, 1)
        noise = (var ** 0.5) * torch.randn(prototype_initials.size()).to(self.device)
        prototype_initials = prototype_initials + noise  # Add Gaussian noize
        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # # Init with Xavier
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization

# Fuann's implementation
class LevelEstimaterSContrastive2(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels,
                 num_prototypes,
                 alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer,
                 args):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer, args)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(args.dropout_rate)

        #self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size, max_norm=1)
        self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size)
        self.init_prototypes = args.init_prototypes
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        # nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal
        self.slv_w = nn.Parameter(torch.tensor(10.0))
        self.slv_b = nn.Parameter(torch.tensor(-5.0))

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        
        assert args.use_layernorm == False            
        if args.loss_type != "cross_entropy":
            if self.with_loss_weight:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs, loss_weights=loss_weights) 
            else:
                self.loss_fct = eval(args.loss_type)(CEFR_lvs=args.CEFR_lvs) 

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs = self.dropout(outputs)
        outputs = mean_pooling(outputs, attention_mask=batch['attention_mask']) 
        
        # positive: compute cosine similarity
        outputs = torch.nn.functional.normalize(outputs)
        positive_prototypes = torch.nn.functional.normalize(self.prototype.weight)
        logits = torch.mm(outputs, positive_prototypes.T)
        logits = self.slv_w * logits + self.slv_b
        logits = logits.reshape((-1, self.num_prototypes, self.CEFR_lvs))
        logits = logits.mean(dim=1)

        # prediction
        predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'labels' in batch:
            labels = batch['labels'].detach().clone()
            # cross-entropy loss
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions
    
    def on_train_start(self) -> None:
        # Init with BERT embeddings
        if self.init_prototypes == "pretrained":
            print("init prototypes from pretrained")
        else:
            print("init prototypes", self.init_prototypes)
            return

        epcilon = 1.0e-6
        labels = []
        prototype_initials = torch.full((self.CEFR_lvs, self.lm.config.hidden_size), fill_value=epcilon).to(self.device)

        self.lm.eval()
        for batch in tqdm.tqdm(self.train_dataloader(), leave=False, desc='init prototypes'):
            labels += batch['labels'].squeeze(-1).detach().clone().numpy().tolist()
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                outputs_mean = mean_pooling(outputs.hidden_states[self.lm_layer],
                                            attention_mask=batch['attention_mask'])
            for lv in range(self.CEFR_lvs):
                prototype_initials[lv] += outputs_mean[
                    batch['labels'].squeeze(-1) == lv].sum(0)
        if not self.with_ib:
            self.lm.train()

        labels = torch.tensor(labels)
        for lv in range(self.CEFR_lvs):
            denom = torch.count_nonzero(labels == lv) + epcilon
            prototype_initials[lv] = prototype_initials[lv] / denom

        var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
        # prototype_initials = torch.repeat_interleave(prototype_initials, self.num_prototypes, dim=0)
        prototype_initials = prototype_initials.repeat(self.num_prototypes, 1)
        noise = (var ** 0.5) * torch.randn(prototype_initials.size()).to(self.device)
        prototype_initials = prototype_initials + noise  # Add Gaussian noize
        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # # Init with Xavier
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
