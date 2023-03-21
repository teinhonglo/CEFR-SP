import torch, transformers
import numpy as np
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) 
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModel, 
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model, 
    HubertModel, 
    WavLMModel,
)
import torchaudio
from sklearn.metrics import f1_score
from typing import Optional, Tuple, Union
from util_w2v import (
    token_embeddings_filtering_padding, 
    read_corpus, 
    CEFRWavDataset, 
    eval_multiclass, 
)
from transformers.pytorch_utils import torch_int_div

class LevelEstimaterW2vBase(pl.LightningModule):
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
        #self.lm_layer = lm_layer
        self.lm_layer = -1
        self.score_name = args.score_name
        # wav2vec2
        self.max_seq_length = args.max_seq_length
        self.wav_feature_extractor_name = args.wav_feature_extractor_name
        self.wav_model_cache_dir = args.wav_model_cache_dir
        self.wav_model_path_or_name = args.wav_model_path_or_name
        self.wav_model_type = args.wav_model_type
        self.use_weighted_layer_sum = False
        self.target_sample_rate = 16000
        self.max_second = args.max_second

        # Load pre-trained model
        self.load_pretrained_lm()
        self.freeze_feature_encoder()
        
    def load_pretrained_lm(self):
        if self.wav_model_type == 'wav2vec2':
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.wav_feature_extractor_name,
                return_attention_mask=True
            )
            self.config = AutoConfig.from_pretrained(self.wav_model_path_or_name)
            self.wav_lm = Wav2Vec2Model(self.config)
            
            num_layers = self.config.num_hidden_layers + 1  # transformer layers + input embeddings
            
            if self.use_weighted_layer_sum:
                self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
                
            # self.projector = nn.Linear(self.config.hidden_size, self.config.hidden_size)
                
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav_lm.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav_lm.parameters():
            param.requires_grad = False                

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths
        
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
            
    def precompute_loss_weights(self, epsilon=1e-5):
        train_levels, _ = read_corpus(self.corpus_path + '/train.tsv', self.num_labels, self.score_name)
         
        train_sentlv_ratio = np.array([np.sum(train_levels == lv) for lv in range(self.CEFR_lvs)])
        train_sentlv_ratio = train_sentlv_ratio / np.sum(train_sentlv_ratio)
        train_sentlv_weights = np.power(train_sentlv_ratio, self.alpha) / np.sum(
            np.power(train_sentlv_ratio, self.alpha)) / (train_sentlv_ratio + epsilon)
        print("loss weight", train_sentlv_weights)
        return torch.Tensor(train_sentlv_weights)

    def encode(self, batch):        
        outputs = self.wav_lm(batch['input_values'], 
                              attention_mask=batch['attention_mask'],                       
                              output_hidden_states=self.config.use_weighted_layer_sum)
        
        _HIDDEN_STATES_START_POSITION = 2
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # hidden_states = self.projector(hidden_states)
        if batch['attention_mask'] is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], batch['attention_mask'])
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            
        return pooled_output, None
        
    def forward(self, inputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def evaluation(self, outputs, test=False):
        pred_labels, gold_labels = [], []
        for output in outputs:
            gold_labels += output['gold_labels'].tolist()
            pred_labels += output['pred_labels'].tolist()

        gold_labels = np.array(gold_labels)
        pred_labels = np.array(pred_labels)
        
        eval_score = f1_score(gold_labels, pred_labels, average='macro')
        logs = {"score": eval_score}

        if test:
            eval_multiclass(self.logger.log_dir + '/sentence', gold_labels, pred_labels)
            with open(self.logger.log_dir + '/test_predictions.txt', 'w') as fw:
                fw.write('Sentence_Lv\n')
                for sent_lv in pred_labels:
                    fw.write('{0}\n'.format(sent_lv))
            
            with open(self.logger.log_dir + '/predictions.txt', 'w') as file:
                predictions_info = '\n'.join(['{} | {}'.format(str(pred[0]), str(target[0])) for pred, target in zip(pred_labels, gold_labels)])
                file.write(predictions_info)

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
            self.corpus_path + '/train.tsv', self.num_labels, self.score_name) 
        #self.train_audio_list = {"audio": [path for path in self.train_info["wav_paths"]]}
        self.train_audio_list = {"input_values": [self.speech_file_to_array_fn(path) for path in self.train_info["wav_paths"]]}
        
        self.dev_levels, self.dev_info = read_corpus(
            self.corpus_path + '/valid.tsv', self.num_labels, self.score_name)
        #self.dev_audio_list = {"audio": [path for path in self.dev_info["wav_paths"]]}
        self.dev_audio_list = {"input_values": [self.speech_file_to_array_fn(path) for path in self.dev_info["wav_paths"]]}
        
        self.test_levels, self.test_info = read_corpus(
            self.test_corpus_path + '/test.tsv', self.num_labels, self.score_name)
        #self.test_audio_list = {"audio": [path for path in self.test_info["wav_paths"]]}
        self.test_audio_list = {"input_values": [self.speech_file_to_array_fn(path) for path in self.test_info["wav_paths"]]}

    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sample_rate)
        speech = resampler(speech_array).squeeze().numpy().astype(np.float32)
        input_values = self.feature_extractor(speech, sampling_rate=self.target_sample_rate).input_values[0]
        return input_values
    
    # return the dataloader for each split
    def train_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent = torch.tensor(self.train_levels, dtype=data_type).unsqueeze(1)
        #inputs = self.my_tokenize(self.train_audio_list)
        
        return DataLoader(
                CEFRWavDataset(self.train_audio_list, y_sent), 
                batch_size=self.batch_size, 
                shuffle=True, 
                collate_fn=self.my_collect_fn)

    def val_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent = torch.tensor(self.dev_levels, dtype=data_type).unsqueeze(1)
        #inputs = self.my_tokenize(self.dev_audio_list)

        return DataLoader(
                CEFRWavDataset(self.dev_audio_list, y_sent), 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=self.my_collect_fn)

    def test_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent = torch.tensor(self.test_levels, dtype=data_type).unsqueeze(1)
        #inputs = self.my_tokenize(self.test_audio_list)

        return DataLoader(
                CEFRWavDataset(self.test_audio_list, y_sent), 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=self.my_collect_fn)

    def my_collect_fn(self, batch):
        max_seq_length = self.max_seq_length
        input_features = [{"input_values": b["input_values"]} for b in batch]
        labels = [b["labels"] for b in batch]
        
        if self.max_second == -1:
            inputs = self.feature_extractor.pad(input_features, 
                                                padding=True,
                                                pad_to_multiple_of=None,
                                                return_tensors="pt")        
        else:
            inputs = self.feature_extractor.pad(input_features, 
                                                padding='max_length',
                                                max_length=self.target_sample_rate * self.max_second,
                                                truncation=True,
                                                pad_to_multiple_of=None,
                                                return_tensors="pt")
        inputs["labels"] = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
        
        return inputs
