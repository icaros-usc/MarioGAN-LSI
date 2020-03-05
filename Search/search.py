import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from util import SearchHelper
from util import bc_calculate
import pathlib
os.environ['CLASSPATH']=os.path.join(pathlib.Path().absolute(),"Mario.jar")
#os.environ['CLASSPATH'] = "/home/tehqin/Projects/MarioGAN-LSI/Mario.jar"


import pandas as pd
import numpy as np
from numpy.linalg import eig
import torch
#import torchvision.utils as vutils
from torch.autograd import Variable

import toml
import json
import numpy
import util.models.dcgan as dcgan
import cma
import torch
#import torchvision.utils as vutils
from torch.autograd import Variable
import json
import numpy
import util.models.dcgan as dcgan
import math
import random
from collections import OrderedDict
import csv
from algorithms import *
from util.SearchHelper import *

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()
"""

if not os.path.exists("logs"):
    os.mkdir("logs")

global EliteMapConfig
EliteMapConfig=[]




def eval_mario(ind,visualize):
    realLevel=to_level(ind.level)
    JString = autoclass('java.lang.String')
    agent = Agent()
    game = MarioGame()
    result = game.runGame(agent, JString(realLevel), 20, 0, visualize)
    #print(result)
    messageReceived=str(result.getCompletionPercentage())+","
    messageReceived+=str(result.getNumJumps())+","
    messageReceived+=str(result.getKillsTotal())+","
    messageReceived+=str(result.getCurrentLives())+","
    messageReceived+=str(result.getNumCollectedTileCoins())+","
    messageReceived+=str(result.getRemainingTime())
    
    
    #messageReceived=sys.stdin.readline()
    statsList=messageReceived.split(',')
    ind.statsList=statsList
    ind.features=[]
    for bc in EliteMapConfig["Map"]["Features"]:
        get_feature=bc["name"]
        get_feature=getattr(bc_calculate,get_feature)
        feature_value=get_feature(ind,result)
        ind.features.append(feature_value)
    ind.features=tuple(ind.features)
    completion_percentage=float(statsList[0])

    return completion_percentage

evaluate = eval_mario



def run_trial(num_to_evaluate,algorithm_name,algorithm_config,elite_map_config,trial_name,model_path,visualize):
    feature_ranges=[]
    column_names=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)']
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"],bc["high"]))
        column_names.append(bc["name"])
    feature_map = FeatureMap(num_to_evaluate, feature_ranges)

    if algorithm_name=="CMAES":
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        algorithm_instance=CMA_ES_Algorithm(num_to_evaluate,mutation_power,population_size,feature_map,trial_name,column_names)
    elif algorithm_name=="CMAME":
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        algorithm_instance=CMA_ME_Algorithm(mutation_power,num_to_evaluate,population_size,feature_map,trial_name,column_names)
    elif algorithm_name=="MAPELITES":
        mutation_power=algorithm_config["mutation_power"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=MapElitesAlgorithm(mutation_power, initial_population, num_to_evaluate, feature_map,trial_name,column_names)
    elif algorithm_name=="ISOLINEDD":
        mutation_power1=algorithm_config["mutation_power1"]
        mutation_power2=algorithm_config["mutation_power2"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=ISOLineDDAlgorithm(mutation_power1, mutation_power2,initial_population, num_to_evaluate, feature_map,trial_name,column_names)
    
    simulation=1
    while algorithm_instance.is_running():
        ind = algorithm_instance.generate_individual(model_path)
        ind.fitness = evaluate(ind,visualize)
        algorithm_instance.return_evaluated_individual(ind)
        print(str(simulation)+"/"+str(num_to_evaluate)+" simulations finished")
        simulation=simulation+1
    algorithm_instance.allRecords.to_csv("logs/"+trial_name+"_all_simulations.csv")

"""
if __name__ == '__main__':
    print("READY") # Java loops until it sees this special signal
    #sys.stdout.flush() # Make sure Java can sense this output before Python blocks waiting for input 
    experiment_toml=toml.load(opt.config)
    num_trials=experiment_toml["num_trials"]
    trial_toml=toml.load(experiment_toml["trial_config"])
    for t in range (num_trials):
        NumSimulations=trial_toml["num_simulations"]
        AlgorithmToRun=trial_toml["algorithm"]
        AlgorithmConfig=toml.load(trial_toml["algorithm_config"])
        EliteMapConfig=toml.load(trial_toml["elite_map_config"])
        TrialName=trial_toml["trial_name"]+str(t+1)
        run_trial(NumSimulations,AlgorithmToRun,AlgorithmConfig,EliteMapConfig,TrialName)
        #below needs to be changed
        print("Finished One Trial")
    print("Finished All Trials")
"""

def start_search(trial_index,experiment_toml,model_path,visualize):
    experiment_toml=experiment_toml["Trials"][trial_index]
    trial_toml=toml.load(experiment_toml["trial_config"])
    NumSimulations=trial_toml["num_simulations"]
    AlgorithmToRun=trial_toml["algorithm"]
    AlgorithmConfig=toml.load(trial_toml["algorithm_config"])
    global EliteMapConfig
    EliteMapConfig=toml.load(trial_toml["elite_map_config"])
    TrialName=trial_toml["trial_name"]+str(trial_index)
    run_trial(NumSimulations,AlgorithmToRun,AlgorithmConfig,EliteMapConfig,TrialName,model_path,visualize)
    print("Finished One Trial")
	
