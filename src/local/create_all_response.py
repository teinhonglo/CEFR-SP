import argparse
import random
import logging
import os
import csv
import sys
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

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


def append_all_response(path, score_names=["content", "pronunciation", "vocabulary"]):
    
    infos = []
    spk_infos = defaultdict(dict)
    header = ""
    lines = _read_tsv(path)
    
    for i, line in enumerate(lines):
        #infos.append("\t".join(line))
        
        if i == 0:
            infos.append("\t".join(line))
            columns = {key:header_index for header_index, key in enumerate(line)}
            continue
        
        wav_path = line[columns['wav_path']]
        text_id = line[columns["text_id"]]
        text = line[columns['text']]
        
        info = text_id.split("_")
        spk_id = "_".join(info[:4])
        
        if spk_id not in spk_infos:
            # 取第一個
            spk_infos[spk_id] = { sn: str(columns[sn]) for sn in score_names }
            spk_infos[spk_id]["wav_path"] = [ wav_path ]
            spk_infos[spk_id]["text"] = [ text ]
        else:
            spk_infos[spk_id]["wav_path"].append(wav_path)
            spk_infos[spk_id]["text"].append(text)
        
    
    for spk_id in spk_infos:
        line = []
        wav_path = "|".join(spk_infos[spk_id]["wav_path"])
        text = ". ".join(spk_infos[spk_id]["text"])
        
        line.append(spk_id)
        line.append(wav_path)
        line.append(text)
        line.append(spk_infos[spk_id]["content"])
        line.append(spk_infos[spk_id]["pronunciation"])
        line.append(spk_infos[spk_id]["vocabulary"])
        infos.append("\t".join(line))
        
    return infos

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="../data-speaking/teemi-tb1p1/0317_trans_stt_tov/1",
                    type=str)

parser.add_argument("--new_data_dir",
                    default="../data-speaking/teemi-tb1p1/0317_trans_stt_tov_ar/1",
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
new_data_dir = args.new_data_dir

if not os.path.isdir(new_data_dir):
    os.makedirs(new_data_dir)

train_infos = append_all_response(os.path.join(data_dir, "train.tsv"))
train_tsv = "\n".join(train_infos)
valid_infos = append_all_response(os.path.join(data_dir, "valid.tsv"))
valid_tsv = "\n".join(valid_infos)
test_infos = append_all_response(os.path.join(data_dir, "test.tsv"))
test_tsv = "\n".join(test_infos)

with open(os.path.join(new_data_dir, "train.tsv"), "w") as fn:
    fn.write(train_tsv)

with open(os.path.join(new_data_dir, "valid.tsv"), "w") as fn:
    fn.write(valid_tsv)

with open(os.path.join(new_data_dir, "test.tsv"), "w") as fn:
    fn.write(test_tsv)
