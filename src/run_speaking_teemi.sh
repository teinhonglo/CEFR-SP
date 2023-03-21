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
alpha=0.5
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
dropout_rate=0.1
lm_layer=11
data_prefix=

extra_options=""

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`
corpus_dir=${corpus_dir}/tb${test_book}

data_dir=../data-speaking/teemi-tb${test_book}p${part}/${data_prefix}${trans_type}
exp_root=../exp-speaking/teemi-tb${test_book}p${part}/${data_prefix}${trans_type}

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

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
    for sn in $score_names; do
        for fd in $folds; do
            echo "$part $sn $fd $exp_tag"
            exp_dir=$exp_tag/$sn/$fd
            if [ -d $exp_root/$exp_tag/$sn/$fd/version_1 ]; then
                echo "$exp_root/$exp_tag/$sn/$fd/version_1 is already existed. Exit!" 
                continue
            else
                rm -rf $exp_root/$exp_tag/$sn/$fd/version_0
            fi
            
            python level_estimator.py --model $model_path --lm_layer $lm_layer $extra_options \
                                      --CEFR_lvs  $max_score \
                                      --seed 985 --num_labels $max_score \
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

            echo "$part $sn $fd"
            echo $checkpoint_path
            exp_dir=$exp_tag/$sn/$fd
            python level_estimator.py --model $model_path --lm_layer $lm_layer $extra_options --do_test \
                                      --CEFR_lvs  $max_score \
                                      --seed 985 --num_labels $max_score \
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
    python local/speaking_predictions_to_report.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --version_dir version_1 \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report.log
    
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --all_bins "$all_bins" \
                                    --cefr_bins "$cefr_bins" \
                                    --scores "$score_names"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then  
    runs_root=$exp_root
    python local/speaking_predictions_to_report_spk.py  --merged_speaker --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --question_type tb${test_book}p${part} \
                                                    --version_dir version_1 \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report_spk.log
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then  
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --all_bins "$all_bins" \
                                    --cefr_bins "$cefr_bins" \
                                    --affix "_spk" \
                                    --scores "$score_names"
fi
