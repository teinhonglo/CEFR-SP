import json
import numpy as np
import torch
from model import LevelEstimaterClassification, LevelEstimaterContrastive, LevelEstimaterContrastiveSED, LevelEstimaterContrastiveDot
from baseline import BaselineClassification

class Grader(object):
    def __init__(self, pretrained_path):
        checkpoint = torch.load(pretrained_path)
        hyper_parameters = checkpoint["hyper_parameters"]
        model_weights = checkpoint["state_dict"]
        
        args = hyper_parameters['args']
        args.pretrained = pretrained_path

        if args.type == "classification":
            model_clsname = "LevelEstimaterClassification"
        elif args.type  == "constrastive":
            model_clsname = "LevelEstimaterContrastive"
        elif args.type  == "constrastive_sed":
            model_clsname = "LevelEstimaterContrastive"
        elif args.type  == "constrastive_dot":
            model_clsname = "LevelEstimaterContrastive"
        
        self.model = eval(model_clsname).load_from_checkpoint(args.pretrained, corpus_path=args.data,
                                                            test_corpus_path=args.test,
                                                            pretrained_model=args.model,
                                                            problem_type=args.type,
                                                            with_ib=args.with_ib,
                                                            with_loss_weight=args.with_loss_weight,
                                                            attach_wlv=args.attach_wlv,
                                                            num_labels=args.num_labels,
                                                            word_num_labels=args.word_num_labels,
                                                            num_prototypes=args.num_prototypes,
                                                            alpha=args.alpha, 
                                                            ib_beta=args.ib_beta,
                                                            batch_size=args.batch,
                                                            learning_rate=args.init_lr,
                                                            warmup=args.warmup, 
                                                            lm_layer=args.lm_layer, 
                                                            args=args)

        self.model.load_state_dict(model_weights)
        self.model.eval()

    def assessing(self, transcript):
        batch = self.model.my_tokenize([transcript.split()])

        with torch.no_grad():
            y_hat = self.model(batch).squeeze(-1)

        return y_hat
