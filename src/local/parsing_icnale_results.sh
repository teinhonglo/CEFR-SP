#!/bin/bash
exp_root=../exp-speaking/icnale/smil_trans_stt_whisper_large
special_tag=""

. ./parse_options.sh

echo "$exp_root and $special_tag"
summ_fn=$exp_root/RESULTS.md
rm -rf $summ_fn

for report_fn in `find $exp_root -name report.log`; do
    model_tag=`dirname $report_fn | xargs basename`
    is_match=`echo $model_tag | grep -E "$special_tag"`

    if [ -z $is_match ]; then
        continue
    else
        echo $model_tag
    fi

    metrics="| exp_tag | `tail -n+4 $report_fn | head -n -1 | awk '{print $1}' | sed -z "s/\n/ | /g"`"
    values="| $model_tag | `tail -n+4 $report_fn | head -n -1 | awk '{print $2}' | sed -z "s/\n/ | /g"`"
    sep=`echo $metrics | sed "s/[A-Za-z0-9_\.]\+/---/g"`
    
    if [ ! -f $summ_fn ]; then
        echo "$metrics" > $summ_fn 
        echo $sep >> $summ_fn
    fi
    echo "$values" >> $summ_fn
done

wc -l $summ_fn
