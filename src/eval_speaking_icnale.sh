#!/bin/bash
# Train from scratch
# level_estimator_classification_lcase_normcls_bert-base-uncased_val_score-max_b8g1_lr5.0e-5_drop0.1
stage=0
stop_stage=1000
# data-related
score_names="holistic"
kfold=1
test_on_valid="true"
trans_type="trans_stt"
do_round="true"
# model-related
model=bert
exp_tag=
model_path=bert-base-uncased
#exp_tag=deberta-model
#model_path=microsoft/deberta-v3-large
max_score=5
max_seq_length=256
max_epochs=-1
alpha=0.5
num_prototypes=3
init_prototypes="pretrained"
monitor="val_score"
monitor_mode="max"
stt_model_name=whisperv2_large
model_type=classification
do_loss_weight=true
do_lower_case=true
init_lr=5.0e-5
batch_size=8
accumulate_grad_batches=1
use_prediction_head=false
use_pretokenizer=false
use_layernorm=false
normalize_cls=true
loss_type="cross_entropy"
dropout_rate=0.1

extra_options=""

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`

data_dir=../data-speaking/icnale/${trans_type}_${stt_model_name}
exp_root=../exp-speaking/icnale/${trans_type}_${stt_model_name}

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
            #python deploy2.py --pretrained $checkpoint_path
            python /share/nas165/teinhonglo/github_repo/grading-system/local/graders/cefr_sp/deploy2.py --pretrained $checkpoint_path
        done
    done 
fi

