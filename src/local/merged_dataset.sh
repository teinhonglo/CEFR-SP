
data_root=../data-speaking

fst_dname=teemi-tb1p1
dnames="teemi-tb1p2 teemi-tb2p1 teemi-tb2p3"
exp_cond="trans_stt_tov"

dest_dname="teemi-tb1-2"

folds=`seq 1 5`

for fd in $folds; do
    dest_dir=$data_root/$dest_name/$exp_cond/$fd
    if [ ! -d $dest_dir ]; then
        mkdir -p $dest_dir
    fi
    
    src_dir=$data_root/$fst_dname/$exp_cond/$fd
    for f in train.tsv valid.tsv test.tsv; do
        cat $src_dir/$f > $dest_dir/$f
    done
    
    for dname in $dnames; do
        src_dir=$data_root/$dname/$exp_cond/$fd
        
        for f in train.tsv valid.tsv test.tsv; do
            tail -n +2 $src_dir/$f >> $dest_dir/$f
        done
    done
done
