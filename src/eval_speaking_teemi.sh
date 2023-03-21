#!/bin/bash
# Train from scratch
stage=0
stop_stage=1000
# data-related
corpus_dir="../corpus/speaking/teemi_pretest"
score_names="content pronunciation vocabulary"
anno_fn="annotation_multi_en_mct_cnn_tdnnf_tgt3meg-dl.xlsx"
kfold=5
test_on_valid="true"
trans_type="trans_stt"
do_round="true"
# model-related
model=bert
exp_tag=
model_path=bert-base-uncased
max_score=8
max_seq_length=128
max_epochs=-1
alpha=0.2
num_prototypes=3
init_prototypes="pretrained"
monitor="val_score"
monitor_mode="max"
model_type=contrastive
do_loss_weight=true
do_lower_case=true
init_lr=5.0e-5
batch_size=8
accumulate_grad_batches=4
use_prediction_head=false
use_pretokenizer=false
use_layernorm=false
normalize_cls=false
loss_type="cross_entropy"
test_book=1
part=1 # 1 = 基礎聽答, 2 = 情境式提問與問答, 3 = 主題式口說任務, 4 = 摘要報告 (不自動評分) 
do_split=true
do_dig=true
ori_all_bins="1,2,2.5,3,3.5,4,4.5,5"
all_bins="1.5,2.5,3.5,4.5,5.5,6.5,7.5"
cefr_bins="1.5,3.5,5.5,7.5"
dropout_rate=0.0

extra_options=""

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`
corpus_dir=${corpus_dir}/tb${test_book}

data_dir=../data-speaking/teemi-tb${test_book}p${part}/${trans_type}
exp_root=../exp-speaking/teemi-tb${test_book}p${part}/${trans_type}

if [ "$test_on_valid" == "true" ]; then
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
fi

if [ "$do_dig" == "true" ]; then
    # [0, 1, 1.5, 2, 2.78, 3.5, 4, 4.25, 5, 4.75] -> [0, 1, 2, 3, 4, 6, 7, 7, 9, 8]
    echo "digitalized"
else
    data_dir=${data_dir}_wod
    exp_root=${exp_root}_wod
fi

if [ "$do_split" == "true" ]; then
    # 一個音檔當一個
    echo "do split"
else
    data_dir=${data_dir}_nosp
    exp_root=${exp_root}_nosp
fi

if [ "$model_type" == "classification" ] || [ "$model_type" == "regression" ]; then
    exp_tag=${exp_tag}level_estimator_${model_type}
else
    if [ $init_prototypes == "pretrained" ]; then
        exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}
    else
        extra_options="$extra_options --init_prototypes ${init_prototypes}"
        exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}_${init_prototypes}
    fi
fi

if [ "$do_loss_weight" == "true" ]; then
    exp_tag=${exp_tag}_loss_weight_alpha${alpha}
    extra_options="$extra_options --with_loss_weight"
fi

if [ "$do_lower_case" == "true" ]; then
    exp_tag=${exp_tag}_lcase
    extra_options="$extra_options --do_lower_case"
fi

if [ "$use_prediction_head" == "true" ]; then
    exp_tag=${exp_tag}_phead
    extra_options="$extra_options --use_prediction_head"
fi

if [ "$use_pretokenizer" == "true" ]; then
    exp_tag=${exp_tag}_pretok
    extra_options="$extra_options --use_pretokenizer"
fi

if [ "$use_layernorm" == "true" ]; then
    exp_tag=${exp_tag}_lnorm
    extra_options="$extra_options --use_layernorm"
fi

if [ "$normalize_cls" == "true" ]; then
    exp_tag=${exp_tag}_normcls
    extra_options="$extra_options --normalize_cls"
fi

if [ "$loss_type" != "cross_entropy" ]; then
    exp_tag=${exp_tag}_${loss_type}
    extra_options="$extra_options --loss_type $loss_type"
fi

if [ "$max_epochs" != "-1" ]; then
    exp_tag=${exp_tag}_ep${max_epochs}
fi

if [ "$max_epochs" != "-1" ]; then
    exp_tag=${exp_tag}_ep${max_epochs}
fi

model_name=`echo $model_path | sed -e 's/\//-/g'`
exp_tag=${exp_tag}_${model_name}_${monitor}-${monitor_mode}_b${batch_size}g${accumulate_grad_batches}_lr${init_lr}_drop${dropout_rate}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            # Test a pretrained model
            checkpoint_path=`find $exp_root/$exp_tag/$sn/$fd/version_0 -name *ckpt`
            
            if [ -z $checkpoint_path ]; then
                echo "No such directories, $exp_root/$exp_tag/$sn/$fd/version_0";
                exit 0;
            fi
            
            echo "$sn $fd"
            echo $checkpoint_path
            exp_dir=$exp_tag/$sn/$fd
            python deploy.py --model $model_path --lm_layer 11 $extra_options --do_test \
                             --CEFR_lvs  $max_score \
                             --seed 66 --num_labels $max_score \
                             --max_epochs $max_epochs \
                             --monitor $monitor \
                             --monitor_mode $monitor_mode \
                             --exp_dir $exp_dir \
                             --score_name $sn \
                             --batch $batch_size --warmup 0 \
                             --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                             --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path

        done
    done 
fi

