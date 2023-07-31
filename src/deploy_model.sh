#!/bin/bash

exp_root=../exp-speaking/teemi-tb1p1/0606_trans_stt_tov
exp_tag=level_estimator_contrastive_num_prototypes3_loss_weight_alpha0.2_lcase_sentence-transformers-all-mpnet-base-v2_val_score-max_b8g4_lr5.0e-5_drop0.0
sys_tag=a01
sys_root=/share/nas165/teinhonglo/github_repo/grading-system/models/graders/cefr_sp
analytic_scores="content pronunciation vocabulary"

. ./parse_options.sh

set -euo pipefail

exp_root=`realpath $exp_root`
if [ -d $sys_root/$sys_tag ]; then
    rm -rf $sys_root/$sys_tag
fi
mkdir -p $sys_root/$sys_tag

for as in $analytic_scores; do
    exp_dir=$exp_root/$exp_tag/$as
    tgt_cpth=$sys_root/$sys_tag/${as}.ckpt
        
    src_cpth=
    max_acc_val=0
    ckpt_paths=`ls $exp_dir/*/version_0/checkpoints/*.ckpt`

    
    for cpth in $ckpt_paths; do
        cfn=`basename $cpth`;
        acc_val=0.`echo $cfn | awk -F"=" '{print $NF}' | cut -d"." -f2`
        echo $cpth
        if [ `echo "$acc_val > $max_acc_val" | bc` -eq 1 ]; then
            max_acc_val=$acc_val
            src_cpth=$cpth
        fi
    done
    
    cp -r $src_cpth $tgt_cpth
    echo "$src_cpth $tgt_cpth" >> $sys_root/$sys_tag/log
    echo "" >> $sys_root/$sys_tag/log
done

