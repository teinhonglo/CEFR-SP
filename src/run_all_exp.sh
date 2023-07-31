
for model_type in classification contrastive contrastive_sed scontrastive; do 
    for with_loss_weight in true false; do 
        for dropout_rate in 0.0 0.1; do 
            ./run_speaking_icnale.sh --model_type $model_type \
                                    --with_loss_weight $with_loss_weight \
                                    --dropout_rate $dropout_rate \
                                    --stage 4
        done; 
    done; 
done
