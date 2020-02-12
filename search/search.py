import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from util import SearchHelper
from util import bc_calculate

os.environ['CLASSPATH'] = "E:/6thSemester/Mario-AI-Framework/src"

import matplotlib
matplotlib.use("agg")
#import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import seaborn as sns
import numpy as np
from numpy.linalg import eig
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import toml
import sys
import json
import numpy
import util.models.dcgan as dcgan
import cma
import random
import math
import matplotlib.pyplot as plt
import os
import csv
#os.chdir("./DagstuhlGAN")
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import json
import numpy
import util.models.dcgan as dcgan
#import matplotlib.pyplot as plt
import math
import random
from collections import OrderedDict
import csv
from algorithms import *
from util.SearchHelper import *

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()

if not os.path.exists(success_map):
    os.mkdir(success_map)
if not os.path.exists(records):
    os.mkdir(records)

EliteMapConfig=[]




def eval_mario(ind):
    realLevel=to_level(ind.level)
    agent = Agent()
    game = MarioGame()
    result = game.runGame(agent, realLevel, 20, 0, True)
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
    for bc in EliteMapConfig["Map.Features"]:
        get_feature=bc["name"]
        get_feature=getattr(bc_calculate,get_feature)
        feature_value=get_feature(ind,result)
        ind.features.append(feature_value)
    ind.features=tuple(ind.features)
    completion_percentage=float(statsList[0])

    return completion_percentage

evaluate = eval_mario



def run_trial(num_to_evaluate,algorithm_name,algorithm_config,elite_map_config,trial_name):
    feature_ranges=[]
    for bc in elite_map_config["Map.Features"]:
        feature_ranges.append((bc["low"],bc["high"]))
    feature_map = FeatureMap(num_to_evaluate, feature_ranges)

    if algorithm_name=="CMAES":
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        algorithm_instance=CMA_ES_Algorithm(num_to_evaluate,mutation_power,population_size,feature_map)
    elif algorithm_name=="CMAME":
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        algorithm_instance=CMA_ME_Algorithm(mutation_power,num_to_evaluate,population_size,feature_map)
    elif algorithm_name=="MAPELITES":
        mutation_power=algorithm_config["mutation_power"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=MapElitesAlgorithm(mutation_power, initial_population, num_to_evaluate, feature_map)
    elif algorithm_name=="ISOLINEDD":
        mutation_power1=algorithm_config["mutation_power1"]
        mutation_power2=algorithm_config["mutation_power2"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=ISOLineDDAlgorithm(mutation_power1, mutation_power2,initial_population, num_to_evaluate, feature_map)
    
    while algorithm_instance.is_running():
        ind = algorithm_instance.generate_individual()
        ind.fitness = evaluate(ind)
        if ind.fitness > best:
            best = ind.fitness
        algorithm_instance.return_evaluated_individual(ind)
    algorithm_instance.allRecords.to_csv("logs\\"+trial_name+".csv")

if __name__ == '__main__':
    print("READY") # Java loops until it sees this special signal
    #sys.stdout.flush() # Make sure Java can sense this output before Python blocks waiting for input 
    parsed_toml=toml.load(opt.config)
    NumSimulations=parsed_toml["num_simulations"]
    AlgorithmToRun=parsed_toml["algorithm"]
    AlgorithmConfig=parsed_toml["algorithm_config"]
    EliteMapConfig=parsed_toml["elite_map_config"]
    TrialName=parsed_toml["trial_name"]
    run_trial(NumSimulations,AlgorithmToRun,AlgorithmConfig,EliteMapConfig,TrialName)
    #below needs to be changed
    print("Finished")