#!/bin/bash
exp_root=../exp-speaking/teemi-tb2p3/trans_stt_tov
score_name="content"
special_tag="*"

. ./parse_options.sh

echo "$exp_root and $special_tag"
summ_fn=$exp_root/RESULTS_${score_name}.md
rm -rf $summ_fn

for report_fn in `find $exp_root -name report_spk.log`; do
    model_tag=`dirname $report_fn | xargs basename`
    is_match=`echo $model_tag | grep -E "$special_tag"`

    if [ -z $is_match ]; then
        continue
    else
        echo $model_tag
    fi
    
    if [ "$score_name" == "content" ]; then
        # content
        metrics="| exp_tag | `head -n 11 $report_fn | tail -n+4 | head -n -1 | awk '{print $1}' | sed -z "s/\n/ | /g"`"
        values="| $model_tag | `head -n 11 $report_fn | tail -n+4 | head -n -1 | awk '{print $2}' | sed -z "s/\n/ | /g"`"
        sep=`echo $metrics | sed "s/[A-Za-z0-9_\.]\+/---/g"`
    elif [ "$score_name" == "pronunciation" ]; then
        # pronunciation
        metrics="| exp_tag | `head -n 20 $report_fn | tail -n+13 | head -n -1 | awk '{print $1}' | sed -z "s/\n/ | /g"`"
        values="| $model_tag | `head -n 20 $report_fn | tail -n+13 | head -n -1 | awk '{print $2}' | sed -z "s/\n/ | /g"`"
        sep=`echo $metrics | sed "s/[A-Za-z0-9_\.]\+/---/g"`
    elif [ "$score_name" == "vocabulary" ]; then
        # vocabulary
        metrics="| exp_tag | `head -n 29 $report_fn | tail -n+22 | head -n -1 | awk '{print $1}' | sed -z "s/\n/ | /g"`"
        values="| $model_tag | `head -n 29 $report_fn | tail -n+22 | head -n -1 | awk '{print $2}' | sed -z "s/\n/ | /g"`"
        sep=`echo $metrics | sed "s/[A-Za-z0-9_\.]\+/---/g"`
    fi
    
    
    if [ ! -f $summ_fn ]; then
        echo "$metrics" > $summ_fn 
        echo $sep >> $summ_fn
    fi
    echo "$values" >> $summ_fn
done

wc -l $summ_fn
