import random
import os
import json
import tqdm
import torch, glob, os, argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from model import LevelEstimaterClassification, LevelEstimaterContrastive, LevelEstimaterContrastiveSED, LevelEstimaterContrastiveDot
from baseline import BaselineClassification

parser = argparse.ArgumentParser(description='CEFR level estimator.')
parser.add_argument('--out', help='output directory', type=str, default='../out/')
parser.add_argument('--data', help='dataset', type=str, required=True)
parser.add_argument('--test', help='dataset', type=str, required=True)
parser.add_argument('--num_labels', help='number of attention heads', type=int, default=6)
parser.add_argument('--alpha', help='weighing factor', type=float, default=0.2)
parser.add_argument('--num_prototypes', help='number of prototypes', type=int, default=3)
parser.add_argument('--init_prototypes', help='initializing prototypes', type=str, default="pretrained")
parser.add_argument('--model', help='Pretrained model', type=str, default='bert-base-cased')
parser.add_argument('--pretrained', help='Pretrained level estimater', type=str, default=None)
parser.add_argument('--type', help='Level estimater type', type=str, required=True,
                    choices=['baseline_reg', 'baseline_cls', 'regression', 'classification', 'contrastive', 'contrastive_sed', 'contrastive_dot'])
parser.add_argument('--with_loss_weight', action='store_true')
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument('--lm_layer', help='number of attention heads', type=int, default=-1)
parser.add_argument('--batch', help='Batch size', type=int, default=128)
parser.add_argument('--seed', help='number of attention heads', type=int, default=42)
parser.add_argument('--init_lr', help='learning rate', type=float, default=1e-5)
parser.add_argument('--val_check_interval', help='Number of steps per validation', type=float, default=1.0)
parser.add_argument('--warmup', help='warmup steps', type=int, default=0)
parser.add_argument('--max_epochs', help='maximum epcohs', type=int, default=-1)
parser.add_argument('--CEFR_lvs', help='number of CEFR levels', type=int, default=8)
parser.add_argument('--score_name', help='score_name for predict and train', type=str, default="vocabulary")
parser.add_argument('--monitor', default='val_score', type=str)
parser.add_argument('--monitor_mode', default='max', type=str)
parser.add_argument('--exp_dir', default='', type=str)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--max_seq_length', default=510, type=int)
parser.add_argument('--use_layernorm', action='store_true')
parser.add_argument('--use_prediction_head', action='store_true')
parser.add_argument('--use_pretokenizer', action='store_true')
parser.add_argument('--normalize_cls', action='store_true')
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--loss_type', default='cross_entropy', type=str)
parser.add_argument('--accumulate_grad_batches', default=1, type=int)
##### The followings are unused arguments: You can just ignore #####
parser.add_argument('--beta', help='balance between sentence and word loss', type=float, default=0.5)
parser.add_argument('--ib_beta', help='beta for information bottleneck', type=float, default=1e-5)
parser.add_argument('--word_num_labels', help='number of attention heads', type=int, default=4)
parser.add_argument('--with_ib', action='store_true')
parser.add_argument('--attach_wlv', action='store_true')


####################################################################
a_args = parser.parse_args()
#torch.manual_seed(args.seed)
#np.random.seed(args.seed)
#random.seed(args.seed)
#gpus = torch.cuda.device_count()

if __name__ == '__main__':
    ############## Train Level Estimator ######################
    #save_dir = 'level_estimator_' + args.type
    #if args.with_loss_weight:
    #    save_dir += '_loss_weight'
    #if args.type == 'contrastive':
    #    save_dir += '_num_prototypes' + str(args.num_prototypes)
    #if args.max_epochs != -1:
    #    save_dir += '_max_epochs' + str(args.max_epochs)
    #
    #if args.monitor != 'train_loss':
    #    save_dir += '_' + args.monitor
    # exp_dir = args.exp_dir
    
    checkpoint = torch.load(a_args.pretrained)
    hyper_parameters = checkpoint["hyper_parameters"]
    args = hyper_parameters['args']
    args.pretrained = a_args.pretrained
    model_weights = checkpoint["state_dict"]

    if args.type  == "constrastive":
        model = LevelEstimaterContrastive.load_from_checkpoint(args.pretrained, corpus_path=args.data,
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
    elif args.type  == "classification":
        model = LevelEstimaterClassification.load_from_checkpoint(args.pretrained, corpus_path=args.data,
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
    '''
    model = LevelEstimaterContrastive(corpus_path=args.data, 
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
    '''
    model.load_state_dict(model_weights)
    model.eval()
    batch = model.my_tokenize(["Okay, so with my previous answer, I do agree with this statement. And going along with the internships and getting you ready for life after college, I think it helps you get into a routine. It helps you set a schedule and it helps you manage time. So I think it's a very good thing. And plus it gets students money. So during college, a lot of us, or many of us, don't have a lot of money. So maybe a job is even necessary, even if you didn't really want the experience. You might need it for the money. So having a job gives you a lot of options and it helps you with just moving on ahead."])
    
    with torch.no_grad():
        y_hat = model(batch)
        print(y_hat)
