

for file_path in `find ../data-speaking/teemi-tb*/*stt_tov/ -name *train.tsv`; do
    fd_dir_name=`dirname $file_path`
    fd_name=`basename $fd_dir_name`
    data_dir=`dirname $fd_dir_name`/$fd_name
    new_data_dir=`dirname $fd_dir_name`_oar/$fd_name
    
    echo $data_dir
    echo $new_data_dir
    python local/create_all_response.py --data_dir $data_dir --new_data_dir $new_data_dir
done
