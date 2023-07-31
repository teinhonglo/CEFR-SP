import torch, transformers
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from util import mean_pooling, token_embeddings_filtering_padding, read_corpus, CEFRDataset, eval_multiclass
import stanza
from metrics_np import compute_metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class LevelEstimaterBase(pl.LightningModule):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv,
                 num_labels,
                 word_num_labels, alpha,
                 batch_size,
                 learning_rate, warmup,
                 lm_layer,
                 args):
        super().__init__()
        self.save_hyperparameters()
        self.CEFR_lvs = args.CEFR_lvs

        if attach_wlv and with_ib:
            raise Exception('Information bottleneck and word labels cannot be used together!') 

        self.corpus_path = corpus_path
        self.test_corpus_path = test_corpus_path
        self.pretrained_model = pretrained_model
        self.with_ib = with_ib
        self.attach_wlv = attach_wlv
        self.num_labels = num_labels
        self.word_num_labels = word_num_labels
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.lm_layer = lm_layer
        self.score_name = args.score_name
        self.do_lower_case = args.do_lower_case
        self.max_seq_length = args.max_seq_length
        self.special_tokens_count = 2
        self.corpus = args.corpus
        self.test_fn = args.test_fn

        # Load pre-trained model
        self.load_pretrained_lm()

        if args.freeze_encoder:
            self.freeze_encoder()

    def load_pretrained_lm(self):
        
        if 'roberta' in self.pretrained_model or 'data2vec' in self.pretrained_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, add_prefix_space=True, do_lower_case=self.do_lower_case, max_length=self.max_seq_length, truncation=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, do_lower_case=self.do_lower_case, max_length=self.max_seq_length, truncation=True)
        self.lm = AutoModel.from_pretrained(self.pretrained_model)
    
    def freeze_encoder(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def precompute_loss_weights(self, epsilon=1e-5):
        train_levels, _ = read_corpus(self.corpus_path + '/train.tsv', self.num_labels, self.score_name, self.corpus)

        train_sentlv_ratio = np.array([np.sum(train_levels == lv) for lv in range(self.CEFR_lvs)])
        train_sentlv_ratio = train_sentlv_ratio / np.sum(train_sentlv_ratio)
        train_sentlv_weights = np.power(train_sentlv_ratio, self.alpha) / np.sum(
            np.power(train_sentlv_ratio, self.alpha)) / (train_sentlv_ratio + epsilon)
        print("Loss Weight: ", train_sentlv_weights)
        return torch.Tensor(train_sentlv_weights)

    def encode(self, batch):
        outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        return outputs.hidden_states[self.lm_layer], None
    
    def encode_emds(self, batch):
        return batch["extra_embs"]

    def forward(self, inputs):
        pass

    def step(self, batch):
        loss, predictions, logs, _ = self.forward(batch)
        gold_labels = batch['labels'].cpu().detach().clone().numpy()
        golds_predictions = {'gold_labels': gold_labels,
                             'pred_labels': predictions}
        return loss, logs

    def _shared_eval_step(self, batch):
        loss, predictions, logs, outputs_mean = self.forward(batch)

        gold_labels = batch['labels'].cpu().detach().clone().numpy()
        outputs_mean = outputs_mean.cpu().detach().clone().numpy()
        golds_predictions = {'gold_labels': gold_labels,
                             'pred_labels': predictions,
                             'outputs_mean': outputs_mean}

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
        if self.test_fn == "train.tsv":
            prefix = "train_"
        elif self.test_fn == "valid.tsv":
            prefix = "valid_"
        else:
            prefix = "test_"
        
        logs = self.evaluation(outputs, test=True, prefix=prefix)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})

    def evaluation(self, outputs, test=False, prefix=""):
        pred_labels, gold_labels, outputs_mean = [], [], []
        for output in outputs:
            gold_labels += output['gold_labels'].tolist()
            pred_labels += output['pred_labels'].tolist()
            outputs_mean += output['outputs_mean'].tolist()

        gold_labels = np.array(gold_labels).squeeze(-1)
        pred_labels = np.array(pred_labels).squeeze(-1)
        logs = {}
        compute_metrics(logs, pred_labels, gold_labels, bins=None)
        
        print("\n\n")
        print("predictions:")
        print("{}".format(pred_labels))
        print("labels:")
        print("{}".format(gold_labels))
        print("\n\n")
        print(logs)

        if test:
            eval_multiclass(self.logger.log_dir + '/' + prefix + 'sentence', gold_labels, pred_labels)
            with open(self.logger.log_dir + '/' + prefix + 'test_predictions.txt', 'w') as fw:
                fw.write('Sentence_Lv\n')
                for sent_lv in pred_labels:
                    fw.write('{0}\n'.format(sent_lv))
            
            with open(self.logger.log_dir + '/' + prefix + 'predictions.txt', 'w') as file:
                predictions_info = '\n'.join(['{} | {}'.format(str(pred), str(target)) for pred, target in zip(pred_labels, gold_labels)])
                file.write(predictions_info)
            
            with open(self.logger.log_dir + '/' + prefix + 'embeddings.txt', 'w') as file:
                output_mean_info = '\n'.join([" ".join(map(str, output_mean)) for output_mean in outputs_mean])
                file.write(output_mean_info)
             
            # 建立 TSNE 模型並降維
            # 注意: 只有ICNALE是對的
            #if len(unique_levels) != 5:
            #    return
            levels = ["A2", "B1_1", "B1_2", "B2", "C"]
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            
            with open(self.logger.log_dir + '/' + prefix + 'labels.txt', 'w') as file:
                predictions_info = '\n'.join(levels)
                file.write(predictions_info)
            
            with open(self.logger.log_dir + '/' + prefix + 'entities.txt', 'w') as file:
                entities_info = '\n'.join(['{}\t{}'.format(str(i), str(levels[target])) for i, target in enumerate(gold_labels)])
                file.write(entities_info)
            
            #pca = PCA(n_components=50)
            #X_pca = pca.fit_transform(outputs_mean)
            X_pca = np.array(outputs_mean)
            
            model = TSNE(n_components=2, init="pca", learning_rate="auto", verbose=1, perplexity=30, n_iter=5000, random_state=66)
            tsne_features = model.fit_transform(X_pca)
            plt.clf()

            for i in range(len(levels)):
                lv_idx = np.where(gold_labels == i)
                plt.scatter(tsne_features[lv_idx, 0], tsne_features[lv_idx, 1], c=colors[i], label=levels[i])
            plt.legend(loc="upper right")
            plt.savefig(self.logger.log_dir + '/' + prefix + 'tsne_plot_gold_labels.png')

            plt.clf()
            for i in range(len(levels)):
                lv_idx = np.where(pred_labels == i)
                plt.scatter(tsne_features[lv_idx, 0], tsne_features[lv_idx, 1], c=colors[i], label=levels[i])
            plt.legend()
            plt.savefig(self.logger.log_dir + '/' + prefix + 'tsne_plot_pred_labels.png')
            plt.clf()
            
        return logs

    def configure_optimizers(self):
        optimizer = transformers.AdamW(self.parameters(), lr=self.learning_rate)
        # Warm-up scheduler
        if self.warmup > 0:
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def prepare_data(self):
        self.train_levels, self.train_info = read_corpus(
            self.corpus_path + '/train.tsv', self.num_labels, self.score_name, self.corpus)
        self.train_inputs = {"sents": self.train_info["sents"], "extra_embs": self.train_info["extra_embs"]}
        
        self.dev_levels, self.dev_info = read_corpus(
            self.corpus_path + '/valid.tsv', self.num_labels, self.score_name, self.corpus)
        self.dev_inputs = {"sents": self.dev_info["sents"], "extra_embs": self.dev_info["extra_embs"]}
        
        self.test_levels, self.test_info = read_corpus(
            self.test_corpus_path + '/' + self.test_fn, self.num_labels, self.score_name, self.corpus)
        self.test_inputs = {"sents": self.test_info["sents"], "extra_embs": self.test_info["extra_embs"]}    

    # return the dataloader for each split
    def train_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent = torch.tensor(self.train_levels, dtype=data_type).unsqueeze(1)
        inputs = self.my_tokenize(self.train_inputs["sents"])
        inputs["extra_embs"] = torch.tensor(self.train_inputs["extra_embs"], dtype=torch.float32)
        
        return DataLoader(CEFRDataset(inputs, y_sent), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent = torch.tensor(self.dev_levels, dtype=data_type).unsqueeze(1)
        inputs = self.my_tokenize(self.dev_inputs["sents"])
        inputs["extra_embs"] = torch.tensor(self.dev_inputs["extra_embs"], dtype=torch.float32)

        return DataLoader(CEFRDataset(inputs, y_sent), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent = torch.tensor(self.test_levels, dtype=data_type).unsqueeze(1)
        inputs = self.my_tokenize(self.test_inputs["sents"])
        inputs["extra_embs"] = torch.tensor(self.test_inputs["extra_embs"], dtype=torch.float32)

        return DataLoader(CEFRDataset(inputs, y_sent), batch_size=self.batch_size, shuffle=False)

    def my_tokenize(self, sents):
        max_seq_length = self.max_seq_length
        special_tokens_count = self.special_tokens_count
        
        for sent_idx, sent in enumerate(sents):

            tokens = []
            for i, word in enumerate(sent):
                if len(tokens) >= max_seq_length - special_tokens_count:
                    break
                word_pieces = self.tokenizer.tokenize(word)


                tokens.extend(word_pieces)
                
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
            
            sents[sent_idx] = self.tokenizer.convert_tokens_to_string(tokens).split()
        
        inputs = self.tokenizer(sents, return_tensors="pt", padding=True,
                                is_split_into_words=True,
                                return_offsets_mapping=True)
        return inputs
