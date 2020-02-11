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
#parsed_toml = toml.load("./config/cma_me.tml")

parsed_toml=toml.load(opt.config)

records=parsed_toml["LogPaths"]["recordsCSV"]

if not os.path.exists(success_map):
    os.mkdir(success_map)
if not os.path.exists(records):
    os.mkdir(records)





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
    featureX=get_featureX(ind,result)
    featureY=get_featureY(ind,result)
    ind.features=(featureX,featureY)
    completion_percentage=float(statsList[0])

    return completion_percentage

evaluate = eval_mario


get_featureX=parsed_toml["featureX"]
get_featureY=parsed_toml["featureY"]
get_featureX=getattr(bc_calculate, get_featureX)
get_featureY=getattr(bc_calculate, get_featureY)



def run_cma_es(num_to_evaluate, mutation_power):

    resolution = (parsed_toml["CMAMESetting"]["StartResolution"],parsed_toml["CMAMESetting"]["EndResolution"])
    feature_ranges=[(parsed_toml["CMAMESetting"]["X_Low"],parsed_toml["CMAMESetting"]["X_High"]),(parsed_toml["CMAMESetting"]["Y_Low"],parsed_toml["CMAMESetting"]["Y_High"])]

    feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolution)
    cmaes = CMA_ES_Algorithm(num_to_evaluate,mutation_power,feature_map)

    while cmaes.is_running():
        ind = cmaes.generate_individual()
        ind.fitness = evaluate(ind)
        cmaes.return_evaluated_individual(ind)
    
    #output all records to csv files
    cmaes.allRecords.to_csv(records+"\\AllRecords.csv")


          

def run_cma_me(num_to_evaluate, mutation_power=parsed_toml["CMAMESetting"]["MutationPower"]):
    resolution = (parsed_toml["CMAMESetting"]["StartResolution"],parsed_toml["CMAMESetting"]["EndResolution"])
    feature_ranges=[(parsed_toml["CMAMESetting"]["X_Low"],parsed_toml["CMAMESetting"]["X_High"]),(parsed_toml["CMAMESetting"]["Y_Low"],parsed_toml["CMAMESetting"]["Y_High"])]

    feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolution)
    
    cma_me = CMA_ME_Algorithm(mutation_power, num_to_evaluate, feature_map)

    best = -10 ** 18
    while cma_me.is_running():
        ind = cma_me.generate_individual()
        ind.fitness = evaluate(ind)
        if ind.fitness > best:
            best = ind.fitness
        cma_me.return_evaluated_individual(ind)

    #output all records to csv files
    cma_me.allRecords.to_csv(records+"\\AllRecords.csv")


def run_map_elites(num_to_evaluate, initial_population, mutation_power=parsed_toml["MAPEliteSetting"]["MutationPower"]):
    resolution = (parsed_toml["MAPEliteSetting"]["StartResolution"],parsed_toml["MAPEliteSetting"]["EndResolution"])
    feature_ranges=[(parsed_toml["CMAMESetting"]["X_Low"],parsed_toml["CMAMESetting"]["X_High"]),(parsed_toml["CMAMESetting"]["Y_Low"],parsed_toml["CMAMESetting"]["Y_High"])]
    feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolution)
    

    me = MapElitesAlgorithm(mutation_power, 
                            initial_population, 
                            num_to_evaluate, 
                            feature_map)

    best = -10 ** 18
    while me.is_running():
        ind = me.generate_individual()
        ind.fitness = evaluate(ind)
        if ind.fitness > best:
            best = ind.fitness
        me.return_evaluated_individual(ind)
    me.allRecords.to_csv(records+"\\AllRecords.csv")


def run_ISOLineDD(num_to_evaluate, initial_population, mutation_power1,mutation_power2):
    resolution = (parsed_toml["MAPEliteSetting"]["StartResolution"],parsed_toml["MAPEliteSetting"]["EndResolution"])
    feature_ranges=[(parsed_toml["CMAMESetting"]["X_Low"],parsed_toml["CMAMESetting"]["X_High"]),(parsed_toml["CMAMESetting"]["Y_Low"],parsed_toml["CMAMESetting"]["Y_High"])]
    feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolution)
    

    me = ISOLineDDAlgorithm(mutation_power1,
                            mutation_power2,
                            initial_population, 
                            num_to_evaluate, 
                            feature_map)

    best = -10 ** 18
    while me.is_running():
        ind = me.generate_individual()
        ind.fitness = evaluate(ind)
        if ind.fitness > best:
            best = ind.fitness
        me.return_evaluated_individual(ind)
    me.allRecords.to_csv(records+"\\AllRecords.csv")


if __name__ == '__main__':
    print("READY") # Java loops until it sees this special signal
    #sys.stdout.flush() # Make sure Java can sense this output before Python blocks waiting for input 
    NumSimulations=parsed_toml["NumSimulations"]
    AlgorithmToRun=parsed_toml["Algorithm"]["AlgorithmName"]
    AlgorithmConfig=parsed_toml["Algorithm"]["AlgorithmConfig"]
    parsed_toml=toml.load(AlgorithmConfig)
    
    #below needs to be changed
    print(AlgorithmToRun)
    if(AlgorithmToRun=="CMAES"):
        run_cma_es(NumSimulations,mutation_power=parsed_toml["CMAESSetting"]["MutationPower"])
    if(AlgorithmToRun=="CMAME"):
        run_cma_me(NumSimulations, mutation_power=parsed_toml["CMAMESetting"]["MutationPower"])
    if(AlgorithmToRun=="MAPELITE"):
        run_map_elites(NumSimulations,initial_population=parsed_toml["MAPEliteSetting"]["InitialPopulation"],mutation_power=parsed_toml["MAPEliteSetting"]["MutationPower"])
    if(AlgorithmToRun=="ISOLineDD"):
        run_ISOLineDD(NumSimulations,initial_population=parsed_toml["ISOLineDDSetting"]["InitialPopulation"],mutation_power1=parsed_toml["ISOLineDDSetting"]["MutationPower1"],mutation_power2=parsed_toml["ISOLineDDSetting"]["MutationPower2"])
    print("saved")
