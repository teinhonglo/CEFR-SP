#!/bin/bash
# Train from scratch
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
stt_model_name=whisper_large
model_type=classification
with_loss_weight=true
do_lower_case=true
init_lr=5.0e-5
batch_size=8
accumulate_grad_batches=1
use_layernorm=false
normalize_cls=false
freeze_encoder=false
loss_type="cross_entropy"
dropout_rate=0.1

extra_options=""

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`

data_dir=../data-speaking/icnale/${trans_type}_${stt_model_name}
exp_root=../exp-speaking/icnale/${trans_type}_${stt_model_name}

if [ "$model_type" == "classification" ] || [ "$model_type" == "regression" ] || [ "$model_type" == "corn" ] ; then
    exp_tag=${exp_tag}level_estimator_${model_type}
else
    if [ $init_prototypes == "pretrained" ]; then
        exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}
    else
        extra_options="$extra_options --init_prototypes ${init_prototypes}"
        exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}_${init_prototypes}
    fi
fi

if [ "$with_loss_weight" == "true" ]; then
    exp_tag=${exp_tag}_loss_weight_alpha${alpha}
    extra_options="$extra_options --with_loss_weight"
fi

if [ "$do_lower_case" == "true" ]; then
    exp_tag=${exp_tag}_lcase
    extra_options="$extra_options --do_lower_case"
fi

if [ "$use_layernorm" == "true" ]; then
    exp_tag=${exp_tag}_lnorm
    extra_options="$extra_options --use_layernorm"
fi

if [ "$normalize_cls" == "true" ]; then
    exp_tag=${exp_tag}_normcls
    extra_options="$extra_options --normalize_cls"
fi

if [ "$freeze_encoder" == "true" ]; then
    exp_tag=${exp_tag}_freezeEnc
    extra_options="$extra_options --freeze_encoder"
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

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            echo "$sn $fd"
            old_data_root=$data_dir/$fd
            new_data_root=${data_dir}_bembs/$fd
            model_root=$exp_root/$exp_tag/$sn/$fd
            
            if [ ! -d $new_data_root ]; then
                mkdir -p $new_data_root
            fi
            
            rsync -avP $old_data_root/*.tsv $new_data_root/
            data_root=$new_data_root

            
            python local/added_extra_embs.py --data_root $data_root \
                                             --model_root $model_root
        done
    done
fi


