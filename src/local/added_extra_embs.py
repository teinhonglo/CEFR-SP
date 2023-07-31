import os
import numpy as np
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser(description='create new datasets')
parser.add_argument('--data_root', type=str, default="../data-speaking/icnale/trans_stt_whisper_large/1", help='output directory')
parser.add_argument('--model_root', type=str, default="../exp-speaking/icnale/trans_stt_whisper_large/level_estimator_classification_loss_weight_alpha0.5_lcase_bert-base-uncased_val_score-max_b8g1_lr5.0e-5_drop0.1/holistic/1")

args = parser.parse_args()

data_root = args.data_root
model_root = args.model_root

embed_dir = os.path.join(model_root, "version_1")


for data_name in [ "train", "valid", "test" ]:
    tsv_fn = os.path.join(data_root, data_name + ".tsv")

    embed_fn = os.path.join(embed_dir, data_name + "_embeddings.txt")
    embed_list = []
    
    # bert embed.    
    with open(embed_fn, "r") as fn:
        for line in fn.readlines():
            #line = np.array(line.split()).astype(np.float32)
            line = " ".join(line.split())
            embed_list.append(line)
    # hf embed.
    
    # tsv
    df = pd.read_table(tsv_fn)
    df['bert_embs'] = embed_list

    df.to_csv(tsv_fn, sep="\t", index=False)
