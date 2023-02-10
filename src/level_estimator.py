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
from util import eval_multiclass, read_corpus, convert_numeral_to_eight_levels
from model import LevelEstimaterClassification, LevelEstimaterContrastive
from baseline import BaselineClassification
from model_base import CEFRDataset

parser = argparse.ArgumentParser(description='CEFR level estimator.')
parser.add_argument('--out', help='output directory', type=str, default='../out/')
parser.add_argument('--data', help='dataset', type=str, required=True)
parser.add_argument('--test', help='dataset', type=str, required=True)
parser.add_argument('--num_labels', help='number of attention heads', type=int, default=6)
parser.add_argument('--alpha', help='weighing factor', type=float, default=0.2)
parser.add_argument('--num_prototypes', help='number of prototypes', type=int, default=3)
parser.add_argument('--model', help='Pretrained model', type=str, default='bert-base-cased')
parser.add_argument('--pretrained', help='Pretrained level estimater', type=str, default=None)
parser.add_argument('--type', help='Level estimater type', type=str, required=True,
                    choices=['baseline_reg', 'baseline_cls', 'regression', 'classification', 'contrastive'])
parser.add_argument('--with_loss_weight', action='store_true')
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument('--lm_layer', help='number of attention heads', type=int, default=-1)
parser.add_argument('--batch', help='Batch size', type=int, default=128)
parser.add_argument('--seed', help='number of attention heads', type=int, default=42)
parser.add_argument('--init_lr', help='learning rate', type=float, default=1e-5)
parser.add_argument('--val_check_interval', help='Number of steps per validation', type=float, default=1.0)
parser.add_argument('--warmup', help='warmup steps', type=int, default=0)
parser.add_argument('--max_epochs', help='maximum epcohs', type=int, default=-1)
##### The followings are unused arguments: You can just ignore #####
parser.add_argument('--beta', help='balance between sentence and word loss', type=float, default=0.5)
parser.add_argument('--ib_beta', help='beta for information bottleneck', type=float, default=1e-5)
parser.add_argument('--word_num_labels', help='number of attention heads', type=int, default=4)
parser.add_argument('--CEFR_lvs', help='number of CEFR levels', type=int, default=8)
parser.add_argument('--score_name', help='score_name for predict and train', type=str, default="vocabulary")
parser.add_argument('--with_ib', action='store_true')
parser.add_argument('--attach_wlv', action='store_true')
parser.add_argument('--monitor', default='val_score', type=str)
parser.add_argument('--monitor_mode', default='max', type=str)
parser.add_argument('--exp_dir', default='', type=str)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--max_seq_length', default=510, type=int)
parser.add_argument('--use_layernorm', action='store_true')
parser.add_argument('--use_prediction_head', action='store_true')
parser.add_argument('--use_pretokenizer', action='store_true')
parser.add_argument('--loss_type', default='cross_entropy', type=str)
parser.add_argument('--accumulate_grad_batches', default=1, type=int)

####################################################################
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
gpus = torch.cuda.device_count()

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
    exp_dir = args.exp_dir

    #save_dir += '_' + args.model.replace('../pretrained_model/', '').replace('/', '-')
    #save_dir = os.path.join(save_dir, args.fold_type)
    logger = TensorBoardLogger(save_dir=args.out, name=exp_dir)

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        filename="level_estimator-{epoch:02d}-{" + args.monitor + ":.6f}",
        save_top_k=1,
        mode=args.monitor_mode
    )
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        min_delta=1e-5,
        patience=10,
        verbose=False,
        mode=args.monitor_mode
    )
    # swa_callback = StochasticWeightAveraging(swa_epoch_start=3)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.type in ['baseline_reg', 'baseline_cls']:
        lv_estimater = BaselineClassification(args.data, args.test, args.model, args.type, args.attach_wlv,
                                              args.num_labels,
                                              args.word_num_labels,
                                              1.0,
                                              args.batch,
                                              args.init_lr,
                                              args.warmup,
                                              args.lm_layer,
                                              args)

    elif args.type in ['regression', 'classification']:
        if args.pretrained is not None:
            lv_estimater = LevelEstimaterClassification.load_from_checkpoint(args.pretrained, corpus_path=args.data,
                                                                             test_corpus_path=args.test,
                                                                             pretrained_model=args.model,
                                                                             problem_type=args.type, 
                                                                             with_ib=args.with_ib,
                                                                             with_loss_weight=args.with_loss_weight,
                                                                             attach_wlv=args.attach_wlv,
                                                                             num_labels=args.num_labels,
                                                                             word_num_labels=args.word_num_labels,
                                                                             alpha=args.alpha, 
                                                                             ib_beta=args.ib_beta,
                                                                             batch_size=args.batch,
                                                                             learning_rate=args.init_lr,
                                                                             warmup=args.warmup,
                                                                             lm_layer=args.lm_layer, 
                                                                             args=args)
        else:
            lv_estimater = LevelEstimaterClassification(corpus_path=args.data, 
                                                    test_corpus_path=args.test, 
                                                    pretrained_model=args.model, 
                                                    problem_type=args.type, 
                                                    with_ib=args.with_ib,
                                                    with_loss_weight=args.with_loss_weight, 
                                                    attach_wlv=args.attach_wlv,
                                                    num_labels=args.num_labels,
                                                    word_num_labels=args.word_num_labels,
                                                    alpha=args.alpha, 
                                                    ib_beta=args.ib_beta, 
                                                    batch_size=args.batch,
                                                    learning_rate=args.init_lr,
                                                    warmup=args.warmup,
                                                    lm_layer=args.lm_layer, 
                                                    args=args)

    elif args.type in ['classification_admsoftmax']:
        if args.pretrained is not None:
            lv_estimater = LevelEstimaterClassificationADMSoftmax.load_from_checkpoint(args.pretrained, corpus_path=args.data,
                                                                             test_corpus_path=args.test,
                                                                             pretrained_model=args.model,
                                                                             problem_type=args.type, 
                                                                             with_ib=args.with_ib,
                                                                             with_loss_weight=args.with_loss_weight,
                                                                             attach_wlv=args.attach_wlv,
                                                                             num_labels=args.num_labels,
                                                                             word_num_labels=args.word_num_labels,
                                                                             alpha=args.alpha, 
                                                                             ib_beta=args.ib_beta,
                                                                             batch_size=args.batch,
                                                                             learning_rate=args.init_lr,
                                                                             warmup=args.warmup,
                                                                             lm_layer=args.lm_layer, 
                                                                             args=args)
        else:
            lv_estimater = LevelEstimaterClassificationADMSoftmax(corpus_path=args.data, 
                                                    test_corpus_path=args.test, 
                                                    pretrained_model=args.model, 
                                                    problem_type=args.type, 
                                                    with_ib=args.with_ib,
                                                    with_loss_weight=args.with_loss_weight, 
                                                    attach_wlv=args.attach_wlv,
                                                    num_labels=args.num_labels,
                                                    word_num_labels=args.word_num_labels,
                                                    alpha=args.alpha, 
                                                    ib_beta=args.ib_beta, 
                                                    batch_size=args.batch,
                                                    learning_rate=args.init_lr,
                                                    warmup=args.warmup,
                                                    lm_layer=args.lm_layer, 
                                                    args=args)

    elif args.type == 'contrastive':
        if args.pretrained is not None:
            lv_estimater = LevelEstimaterContrastive.load_from_checkpoint(args.pretrained, corpus_path=args.data,
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
        else:
            lv_estimater = LevelEstimaterContrastive(corpus_path=args.data, 
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
    

    if args.pretrained is not None:
        trainer = pl.Trainer(gpus=gpus, logger=logger)
        trainer.test(lv_estimater)
    else:
        # w/o learning rate tuning
        trainer = pl.Trainer(gpus=gpus, 
                            logger=logger, 
                            val_check_interval=args.val_check_interval, 
                            max_epochs=args.max_epochs,
                            accumulate_grad_batches=args.accumulate_grad_batches, 
                            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor])
        trainer.fit(lv_estimater)

        # automatically loads the best weights for you
        trainer.test(ckpt_path="best")
