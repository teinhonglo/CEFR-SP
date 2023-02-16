import torch, scipy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import sys
from local.nlp_models import NlpModel

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.data_len=len(encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.data_len

class CEFRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, levels):
        self.encodings = encodings
        self.labels = levels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def read_corpus(path, num_labels, score_name):
    ids, levels, sents, wav_paths, extra_embs = [], [], [], [], []
    #nlp_model = NlpModel()

    lines = _read_tsv(path)
    for i, line in enumerate(lines):
        if i == 0:
            columns = {key:header_index for header_index, key in enumerate(line)}
            continue
        
        wav_path_list = line[columns['wav_path']].split(" | ")
        text_list = line[columns['text']].split(" | ") 
       
        for j in range(len(text_list)):
            # remove a leading- or tailing-space of the utterance.
            wav_path = " ".join(wav_path_list[j].split())
            text = " ".join(text_list[j].split()).split()
            #cefr_emb = [nlp_model.vocab_profile_feats(text_list[j])["vp_" + feats_id ] for feats_id in ["a1", "a2", "b1", "b2", "c1", "c2"]]
            
            ids.append(line[columns["text_id"]])
            levels.append(float(line[columns[score_name]]) - 1)  # Convert 1-8 to 0-7
            sents.append(text)
            wav_paths.append(wav_path)
            extra_embs.append(float(line[columns[score_name]]) - 1)

    levels = np.array(levels)

    return levels, {"ids": ids, "sents": sents, "wav_paths": wav_paths, "extra_embs": extra_embs}


def _read_tsv(input_file, quotechar=None):
    print(input_file)
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def convert_numeral_to_eight_levels(levels):
    level_thresholds = np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    return _conversion(level_thresholds, levels)


def _conversion(level_thresholds, values):
    thresh_array = np.tile(level_thresholds, reps=(values.shape[0], 1))
    array = np.tile(values, reps=(1, level_thresholds.shape[0]))
    levels = np.maximum(np.zeros((values.shape[0], 1)),
                        np.count_nonzero(thresh_array <= array, axis=1, keepdims=True) - 1).astype(int)
    return levels


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Take attention mask into account for excluding padding
def token_embeddings_filtering_padding(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return token_embeddings * input_mask_expanded


def eval_multiclass(out_path, labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure()
    sns.heatmap(cm)
    plt.savefig(out_path + '_confusion_matrix.png')

    report = classification_report(labels, predictions, digits=4)
    print(report)
    with open(out_path + '_test_report.txt', 'w') as fw:
        fw.write('{0}\n'.format(report))
    


def mean_confidence_interval(data, confidence=0.95):
    if len(data) > 5:
        data.remove(max(data))
        data.remove(min(data))
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h
