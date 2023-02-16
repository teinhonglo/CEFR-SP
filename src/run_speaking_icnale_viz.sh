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
monitor="val_score"
monitor_mode="max"
stt_model_name=whisperv2_large
model_type=contrastive
do_loss_weight=true
do_lower_case=true
init_lr=5.0e-5
batch_size=32
accumulate_grad_batches=1
use_prediction_head=false
use_pretokenizer=false
loss_type="cross_entropy"
dropout_rate=0.1

extra_options=""

. ./path.sh
. ./parse_options.sh

set -euo pipefail

data_dir=../data-speaking/icnale/${trans_type}_${stt_model_name}
exp_root=../exp-speaking/icnale/${trans_type}_${stt_model_name}
folds=`seq 1 $kfold`

if [ "$model_type" == "classification" ] || [ "$model_type" == "regression" ]; then
    exp_tag=${exp_tag}level_estimator_${model_type}
else
    exp_tag=${exp_tag}level_estimator_${model_type}_num_prototypes${num_prototypes}
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

            test_fn="test.tsv"
            for i in 2 3 4; do
                target_dir=$exp_root/$exp_tag/$sn/$fd/version_${i}
                rm -rf $target_dir
            done
            
            target_dir=$exp_root/$exp_tag/$sn/$fd/version_2
            if [ -d $target_dir ]; then
                rm -rf $target_dir
            fi
            python level_estimator_viz.py --model $model_path --lm_layer 11 $extra_options --do_test \
                                          --CEFR_lvs  $max_score \
                                          --seed 66 --num_labels $max_score \
                                          --max_epochs $max_epochs \
                                          --monitor $monitor \
                                          --monitor_mode $monitor_mode \
                                          --exp_dir $exp_dir \
                                          --score_name $sn \
                                          --batch $batch_size --warmup 0 \
                                          --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                          --test_fn $test_fn \
                                          --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path

            test_fn="valid.tsv"
            target_dir=$exp_root/$exp_tag/$sn/$fd/version_3
            if [ -d $target_dir ]; then
                rm -rf $target_dir
            fi
            python level_estimator_viz.py --model $model_path --lm_layer 11 $extra_options --do_test \
                                          --CEFR_lvs  $max_score \
                                          --seed 66 --num_labels $max_score \
                                          --max_epochs $max_epochs \
                                          --monitor $monitor \
                                          --monitor_mode $monitor_mode \
                                          --exp_dir $exp_dir \
                                          --score_name $sn \
                                          --batch $batch_size --warmup 0 \
                                          --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                          --test_fn $test_fn \
                                          --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path
            test_fn="train.tsv"
            target_dir=$exp_root/$exp_tag/$sn/$fd/version_4
            if [ -d $target_dir ]; then
                rm -rf $target_dir
            fi
            python level_estimator_viz.py --model $model_path --lm_layer 11 $extra_options --do_test \
                                          --CEFR_lvs  $max_score \
                                          --seed 66 --num_labels $max_score \
                                          --max_epochs $max_epochs \
                                          --monitor $monitor \
                                          --monitor_mode $monitor_mode \
                                          --exp_dir $exp_dir \
                                          --score_name $sn \
                                          --batch $batch_size --warmup 0 \
                                          --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                          --test_fn $test_fn \
                                          --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path
            for i in 2 3 4; do
                target_dir=$exp_root/$exp_tag/$sn/$fd/version_${i}
                cp -r $target_dir/*.png $target_dir/*.txt $exp_root/$exp_tag/$sn/$fd/version_1/
                rm -rf $target_dir
            done
        done
    done 
fi

