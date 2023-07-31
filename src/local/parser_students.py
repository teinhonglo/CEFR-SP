import argparse
import random
import logging
import os
import csv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
from metrics_cpu import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument("--result_roots",
                    default="../exp-speaking/teemi-tb1p1/0417_trans_stt_tov,../exp-speaking/teemi-tb1p2/0417_trans_stt_tov",
                    type=str)
 
parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--model",
                    default="level_estimator_contrastive_num_prototypes3_loss_weight_alpha0.2_lcase_sentence-transformers-all-mpnet-base-v2_val_score-max_b8g4_lr5.0e-5_drop0.0",
                    type=str)


args = parser.parse_args()

result_roots = args.result_roots
model = args.model
scores = args.scores.split()

anno_columns = ["anno"]
pred_columns = ["pred"]
read_columns = ["text_id"] + anno_columns + pred_columns
export_columns = ["student_id"]
studid_fn = "all.studid"
student_ids = defaultdict(list)

convert_dict = {
                                0: 0,
                                1: 1,
                                2: 2,
                                3: 2.5,
                                4: 3,
                                5: 3.5,
                                6: 4,
                                7: 4.5,
                                8: 5
                     }

#with open(studid_fn, "r") as fn:
#    for line in fn.readlines():
#        line = line.split()[0]
#        student_ids.append("u" + line)

student_info = defaultdict(dict)

for result_root in result_roots.split(","):
    test_book = result_root.split("/")[2]
    student_info[test_book] = {}
    
    
    for score in scores:
        student_info[test_book][score] = {}
        xlsx_path = os.path.join(result_root, model, score, "kfold_detail_spk.xlsx")
        df = pd.read_excel(xlsx_path, sheet_name="All")
        
        if len(student_ids[test_book]) == 0:
            student_ids[test_book] = df["text_id"].tolist()
        #df = df[df["text_id"].isin(student_ids)]
        export_columns.append(test_book + "_" + score)

        for rc in read_columns:
            student_info[test_book][score][rc] = df[rc].tolist()

# 取所有題本學生的id。
all_student_ids = []
for test_book in list(student_info.keys()):
    all_student_ids += student_ids[test_book]

all_student_ids = list(set(all_student_ids))

export_dict = {ec: [] for ec in export_columns}
student_ids_inv = {sid: i for i, sid in enumerate(all_student_ids)}

for sid in all_student_ids:
    export_dict["student_id"].append(sid)

for test_book in list(student_info.keys()):
    for score in list(student_info[test_book].keys()):
        pred_list = np.zeros(len(all_student_ids))
        col_name = test_book + "_" + score
        
        for i, sid in enumerate(student_info[test_book][score]["text_id"]):
            pred_val = convert_dict[int(student_info[test_book][score]["pred"][i])]
            pred_idx = student_ids_inv[sid]
            pred_list[pred_idx] = pred_val

        pred_list = pred_list.tolist()
        export_dict[col_name] = pred_list

export_df = pd.DataFrame.from_dict(export_dict)
export_df.to_excel("report.xlsx", index=False)
