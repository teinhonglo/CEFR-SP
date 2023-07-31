import pandas as pd
import os
# train(90%)/ valid(5%)/ test(5%)
import random
import argparse

random.seed(66)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root",
                    default="../data-speaking/icnale/trans_stt_whisper_large")

args = parser.parse_args()

data_root = args.data_root
data_root1 = "/".join(data_root.split("/")[:-1])
data_root2 = "smil_" + data_root.split("/")[-1]
new_data_root = os.path.join(data_root1, data_root2, "1")

if not os.path.exists(new_data_root):
    os.makedirs(new_data_root)

test_sets = [ "train.tsv", "valid.tsv", "test.tsv" ]
df_list = []

for test_set in test_sets:
    tsv_fn = os.path.join(data_root, "1", test_set)
    df = pd.read_csv(tsv_fn, sep="\t")
    
    spk_ids = []
    for i in range(len(df["text_id"])):
        text_id_info = df["text_id"][i].split("_")
        spk_id = "_".join([text_id_info[1], text_id_info[3]])
        spk_ids.append(spk_id)
    
    df["spk_id"] = spk_ids
    df_list.append(df)

all_df = pd.concat(df_list)
spk_ids = list(set(all_df["spk_id"].values.tolist()))
num_spks = len(spk_ids)

#first_idx = int(num_spks * 0.9)
#second_idx = first_idx + int(num_spks * 0.05)

#train_spk_ids = spk_ids[:first_idx]
#valid_spk_ids = spk_ids[first_idx: second_idx]
#test_spk_ids = spk_ids[second_idx:]

# 因為ID有國籍資訊，所以可以balanced L1
train_spk_ids = []
valid_spk_ids = []
test_spk_ids = []
l1_list = list(set([ sid.split("_")[0] for sid in spk_ids]))
l1_dict = {l1: [] for l1 in l1_list}

for sid in spk_ids:
    l1, codes = sid.split("_")
    l1_dict[l1].append(sid)

for l1, l1_sids in l1_dict.items():
    random.shuffle(l1_sids)
    num_l1_spks = len(l1_sids)
    first_idx = int(num_l1_spks * 0.9)
    second_idx = first_idx + int(num_l1_spks * 0.05)
    
    train_spk_ids += l1_sids[:first_idx]
    valid_spk_ids += l1_sids[first_idx:second_idx]
    test_spk_ids += l1_sids[second_idx:]

print("Total number of the speaker is {}, where #train: {}, #valid: {}, #test: {}".format(num_spks, len(train_spk_ids), len(valid_spk_ids), len(test_spk_ids)))

train_df = all_df[all_df["spk_id"].isin(train_spk_ids)]
valid_df = all_df[all_df["spk_id"].isin(valid_spk_ids)]
test_df = all_df[all_df["spk_id"].isin(test_spk_ids)]

train_df = train_df.drop(["spk_id"], axis=1)
valid_df = valid_df.drop(["spk_id"], axis=1)
test_df = test_df.drop(["spk_id"], axis=1)

#text_id	wav_path	text	holistic
train_df.to_csv(os.path.join(new_data_root, "train.tsv"), sep="\t", index=False)
valid_df.to_csv(os.path.join(new_data_root, "valid.tsv"), sep="\t", index=False)
test_df.to_csv(os.path.join(new_data_root, "test.tsv"), sep="\t", index=False)
