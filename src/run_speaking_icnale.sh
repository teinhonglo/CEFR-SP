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
with_ib=false
loss_weight_type=1
do_lower_case=true
init_lr=5.0e-5
batch_size=8
accumulate_grad_batches=1
use_layernorm=false
normalize_cls=false
freeze_encoder=false
loss_type="cross_entropy"
dropout_rate=0.1
pretrained_path=
data_prefix=

extra_options=""

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`

data_dir=../data-speaking/icnale/${data_prefix}${trans_type}_${stt_model_name}
exp_root=../exp-speaking/icnale/${data_prefix}${trans_type}_${stt_model_name}

if [ "$model_type" == "classification" ] || [ "$model_type" == "regression" ] || [ "$model_type" == "corn" ] ; then
    exp_tag=${exp_tag}level_estimator_${model_type}
else
    if [ $init_prototypes == "pretrained" ]; then
        exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}
    else
        extra_options="$extra_options --init_prototypes ${init_prototypes}"
        exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}_${init_prototypes}
    fi
    if [ $with_ib == "true" ]; then
        extra_options="$extra_options --with_ib"
        exp_tag=${exp_tag}_with_ib
    fi
fi

if [ "$with_loss_weight" == "true" ]; then
    exp_tag=${exp_tag}_loss_weight_alpha${alpha}
    if [ "$loss_weight_type" != "1" ]; then
        exp_tag=${exp_tag}_loss_type$loss_weight_type
    fi
    extra_options="$extra_options --with_loss_weight --loss_weight_type $loss_weight_type"
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
    exp_ta=g${exp_tag}_normcls
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

if [ ! -z $pretrained_path ]; then
    # ../exp-speaking/icnale/smil_trans_stt_whisper_large/level_estimator_classification_lcase_bert-base-uncased_val_score-max_b32g1_lr5.0e-5_drop0.1/holistic/1/version_0/checkpoints
    checkpoint_path=`find $pretrained_path -name *ckpt`
    
    pr_tag=`basename $pretrained_path`
    exp_tag=${exp_tag}_${model_name}_${monitor}-${monitor_mode}_b${batch_size}g${accumulate_grad_batches}_lr${init_lr}_drop${dropout_rate}_pr$pr_tag
else
    exp_tag=${exp_tag}_${model_name}_${monitor}-${monitor_mode}_b${batch_size}g${accumulate_grad_batches}_lr${init_lr}_drop${dropout_rate}
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            echo "$sn $fd"
            exp_dir=$exp_tag/$sn/$fd
            if [ -d $exp_root/$exp_tag/$sn/$fd/version_1 ]; then
                echo "$exp_root/$exp_tag/$sn/$fd/version_1 is already existed. Exit!" 
                continue
            else
                rm -rf $exp_root/$exp_tag/$sn/$fd/version_0
            fi
            
            train_extra_options=
            if [ ! -z $pretrained_path ]; then
                checkpoint_path=`find $pretrained_path/$sn/$fd/version_0 -name *ckpt`
                train_extra_options="--pretrained $checkpoint_path"
            fi
            
            python level_estimator.py --model $model_path --lm_layer 11 $extra_options $train_extra_options \
                                      --corpus "icnale" \
                                      --CEFR_lvs  $max_score \
                                      --seed 66 --num_labels $max_score \
                                      --max_epochs $max_epochs \
                                      --monitor $monitor \
                                      --monitor_mode $monitor_mode \
                                      --out $exp_root \
                                      --exp_dir $exp_dir \
                                      --score_name $sn \
                                      --batch $batch_size --warmup 0 \
                                      --accumulate_grad_batches $accumulate_grad_batches \
                                      --dropout_rate $dropout_rate \
                                      --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                      --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd 
        done
    done
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            # Test a pretrained model
            checkpoint_path=`find $exp_root/$exp_tag/$sn/$fd/version_0 -name *ckpt`
            
            if [ -z $checkpoint_path ]; then
                echo "No such directories, $exp_root/$exp_tag/$sn/$fd/version_0";
                exit 0;
            fi
            
            if [ -d $exp_root/$exp_tag/$sn/$fd/version_1 ]; then
                rm -rf $exp_root/$exp_tag/$sn/$fd/version_1
            fi

            echo "$sn $fd"
            echo $checkpoint_path
            exp_dir=$exp_tag/$sn/$fd
            python level_estimator.py --model $model_path --lm_layer 11 $extra_options --do_test \
                                      --corpus "icnale" \
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

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    runs_root=$exp_root
    python local/speaking_predictions_to_reportv2.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --folds "$folds" \
                                                    --version_dir version_1 \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report.log
    
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualizationv2.py   --result_root $runs_root/$exp_tag \
                                    --scores "$score_names"
fi
