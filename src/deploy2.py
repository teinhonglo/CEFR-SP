import random
import os
import json
import tqdm
import argparse
from runner import Grader

parser = argparse.ArgumentParser(description='CEFR level estimator.')
parser.add_argument('--pretrained', help='Pretrained level estimater', type=str, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    asr_transcript = "Okay, so with my previous answer, I do agree with this statement. And going along with the internships and getting you ready for life after college, I think it helps you get into a routine. It helps you set a schedule and it helps you manage time. So I think it's a very good thing. And plus it gets students money. So during college, a lot of us, or many of us, don't have a lot of money. So maybe a job is even necessary, even if you didn't really want the experience. You might need it for the money. So having a job gives you a lot of options and it helps you with just moving on ahead."
    holistic_grader = Grader(args.pretrained)
    holistic_score = holistic_grader.assessing(asr_transcript)
    print("holistic_score", holistic_score)
