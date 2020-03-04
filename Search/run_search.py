import toml
import argparse
from search import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('-w','--workerID',help='the workerID of the worker to be called',type=int,required=True)
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()

experiment_toml=toml.load(opt.config)
model_path=experiment_toml["GAN_model_path"]
num_list=[]
workerID=opt.workerID
if workerID < 0:
    print("workerID should be greater than or equal to 0")
else:
    for i in experiment_toml["Trials"]:
        num_list.append(i["num_trials"])
    for trial_index in range(len(num_list)):
        if workerID<num_list[trial_index]:
            start_search(trial_index,experiment_toml,model_path)
            break
        workerID=workerID-num_list[trial_index]
    if trial_index == len(num_list):
        print("workerID is greater than the total number of trials")
    
