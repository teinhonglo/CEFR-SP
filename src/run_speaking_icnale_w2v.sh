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
exp_tag=bert-model
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
init_lr=1.0e-5
batch_size=4
accumulate_grad_batches=2
use_prediction_head=false
dropout_rate=0.1
max_second=60
wav_model_type=wav2vec2
wav_feature_extractor_name="facebook/wav2vec2-base"
wav_model_path_or_name="facebook/wav2vec2-base"
wav_model_cache_dir=

extra_options=""

#. ./path_cu111.sh
. ./path.sh
. ./parse_options.sh

set -euo pipefail

data_dir=../data-speaking/icnale/${trans_type}_${stt_model_name}
exp_root=../exp-speaking/icnale/${trans_type}_${stt_model_name}
folds=`seq 1 $kfold`

if [ "$model_type" == "classification" ] || [ "$model_type" == "regression" ]; then
    exp_tag=level_estimator_w2v_${model_type}
else
    exp_tag=level_estimator_w2v_${model_type}_num_prototypes${num_prototypes}
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

if [ "$max_epochs" != "-1" ]; then
    exp_tag=${exp_tag}_ep${max_epochs}
fi

if [ "$max_second" != "-1" ]; then
    exp_tag=${exp_tag}_ep${max_epochs}_ms${max_second}
fi

model_name=`echo $wav_model_path_or_name | sed -e 's/\//-/g'`
exp_tag=${exp_tag}_${model_name}_${monitor}-${monitor_mode}_b${batch_size}g${accumulate_grad_batches}_lr${init_lr}_drop${dropout_rate}_maxseq${max_seq_length}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            echo "$sn $fd"
            exp_dir=$exp_tag/$sn/$fd
            rm -rf $exp_dir
            python level_estimator_w2v.py --model $model_path --lm_layer 11 $extra_options \
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
                                      --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd \
                                      --max_second $max_second \
                                      --wav_model_type $wav_model_type \
                                      --wav_feature_extractor_name $wav_feature_extractor_name \
                                      --wav_model_path_or_name $wav_model_path_or_name 
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
            python level_estimator_w2v.py --model $model_path --lm_layer 11 $extra_options --do_test \
                                      --CEFR_lvs  $max_score \
                                      --seed 66 --num_labels $max_score \
                                      --max_epochs $max_epochs \
                                      --monitor $monitor \
                                      --monitor_mode $monitor_mode \
                                      --exp_dir $exp_dir \
                                      --score_name $sn \
                                      --batch $batch_size --warmup 0 \
                                      --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                      --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path \
                                      --max_second $max_second \
                                      --wav_model_type $wav_model_type \
                                      --wav_feature_extractor_name $wav_feature_extractor_name \
                                      --wav_model_path_or_name $wav_model_path_or_name

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
