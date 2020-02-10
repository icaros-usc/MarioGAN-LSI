import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from util import helper
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
import sys
import json
import numpy
import util.models.dcgan as dcgan
#import matplotlib.pyplot as plt
import math
import random
from collections import OrderedDict
import csv

from jnius import autoclass
MarioGame = autoclass('engine.core.MarioGame')
Agent = autoclass('agents.robinBaumgarten.Agent')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()
#parsed_toml = toml.load("./config/cma_me.tml")

parsed_toml=toml.load(opt.config)

success_map=parsed_toml["LogPaths"]["success_map"]
records=parsed_toml["LogPaths"]["recordsCSV"]

if not os.path.exists(success_map):
    os.mkdir(success_map)
if not os.path.exists(records):
    os.mkdir(records)

num_params = parsed_toml["ModelParameter"]["num_params"]
boundary_value = parsed_toml["ModelParameter"]["boundary_value"]
batchSize = parsed_toml["ModelParameter"]["batchSize"]
nz = parsed_toml["ModelParameter"]["nz"]  # Dimensionality of latent vector

imageSize = parsed_toml["ModelParameter"]["imageSize"]
ngf = parsed_toml["ModelParameter"]["ngf"]
ngpu = parsed_toml["ModelParameter"]["ngpu"]
n_extra_layers = parsed_toml["ModelParameter"]["n_extra_layers"]

features = parsed_toml["ModelParameter"]["features"]

generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)
generator.load_state_dict(torch.load(parsed_toml["GANModelPath"], map_location=lambda storage, loc: storage))

COIN=parsed_toml["GameSetting"]["COIN"]
GROUND = parsed_toml["GameSetting"]["GROUND"]
ENEMY = parsed_toml["GameSetting"]["ENEMY"]
PIPE = parsed_toml["GameSetting"]["PIPE"]
EMPTY = parsed_toml["GameSetting"]["EMPTY"]

Ground=parsed_toml["HeightSeparator"]["Ground"]
AboveGround=parsed_toml["HeightSeparator"]["AboveGround"]
MiddleLevel=parsed_toml["HeightSeparator"]["MiddleLevel"]
HigherLevel=parsed_toml["HeightSeparator"]["HigherLevel"]

def gan_generate(x):
    latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1,
                                              1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    levels.data = levels.data[:, :, :16, :56]
    im = levels.data.cpu().numpy()
    im = numpy.argmax( im, axis = 1)
    
    ground=im[:,Ground,:]
    above=im[:,0:AboveGround,:]
    higher=im[:,0:HigherLevel,:]
    middle=im[:,MiddleLevel:AboveGround,:]
    """
    num_above_ground=len(above[above!=2])
    higher_level=len(higher[higher!=2])
    middle_level=len(middle[middle!=2])
    num_ground=len(ground[ground==GROUND])
    num_ground_enemies=len(ground[ground==ENEMY])
    """
    num_enemies =  len (im[im == ENEMY])
    num_non_empty=len(higher[higher!=EMPTY])
    #num_non_empty=num_non_empty/2 #range for each cell in x axis is 2
    return json.dumps(im[0].tolist())#,num_non_empty,num_enemies

with open('GANTraining/index2str.json') as f:
  index2str = json.load(f)
def get_char(x):
    return index2str[str(x)] 

def to_level(number_level):
    result = []
    number_level=eval(number_level)
    for x in number_level:
        #print(x)
        result.append(''.join(get_char(y) for y in x)+'\n')
    result= ''.join(result)
    return result

"""
def eval_sphere(vs):
    vs = [x-boundary_value*0.4 for x in vs]
    return -sum([x**2 for x in vs])

def eval_rastrigin(vs):
    vs = [x-boundary_value*0.4 for x in vs]
    A = 10
    left = A * len(vs)
    right = sum([x**2 - A * math.cos(2 * math.pi * x) for x in vs])
    return -(left + right)
"""

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

"""
def calc_higher_level_non_empty_blocks(ind,result):
    im=np.array(json.loads(ind.level))
    higher=im[0:HigherLevel,:]
    num_non_empty=len(higher[higher!=EMPTY])
    #num_non_empty=len(im[im!=EMPTY])
    return num_non_empty

def calc_num_enemies(ind,result):
    im=np.array(json.loads(ind.level))
    num_enemies =  len (im[np.isin(im,ENEMY)])
    return num_enemies

def calc_coins_collected(ind,result):
    return result.getNumCollectedTileCoins()
"""

get_featureX=parsed_toml["featureX"]
get_featureY=parsed_toml["featureY"]
"""
possibles=globals().copy()
get_featureX=possibles.get(get_featureX)
get_featureY=possibles.get(get_featureY)
"""
get_featureX=getattr(bc_calculate, get_featureX)
get_featureY=getattr(bc_calculate, get_featureY)
#get_featureX=Higher_Level_Non_Empty_Blocks
#get_featureY=Num_Enemies


def gaussian():
    u1 = 1.0 - random.random()
    u2 = 1.0 - random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)

def create_image(recordFrame, filepath):
    with sns.axes_style("white"):
        fig = sns.relplot(x='x', y='y', hue='f',
               size='f',
               sizes=(1, 20), alpha=.9, data=recordFrame, palette="Purples")
        fig.set(xlabel='Number of Non-Empty Block', ylabel='Number of Enemies')
        axes = fig.axes

        shift_amount = 0.6*boundary_value*num_params
        lo = -shift_amount
        hi = shift_amount
        axes[0,0].set_xlim(0, 40)
        axes[0,0].set_ylim(0, 20)

        fig.savefig(filepath)
    plt.close('all')

def make_record_frame(points):
    x, y, f = zip(*points)
    d = {'x':x, 'y':y, 'f':f}
    return pd.DataFrame(d)




class Individual:

    def __init__(self):
        pass

    def reduce(self, v):
        if abs(v) > boundary_value:
            return boundary_value / v
        return v

    def make_features(self):
        pass
        #half = len(self.param_vector) // 2
        #capped_vals = [self.reduce(x) for x in self.param_vector]
        #self.features = (float(sum(capped_vals[:half])), 
        #                 float(sum(capped_vals[half:])))
    
    def read_mario_features(self):
        pass


class DecompMatrix:
    def __init__(self, dimension):
        self.C = np.eye(dimension, dtype=np.float_) 
        self.eigenbasis = np.eye(dimension, dtype=np.float_)
        self.eigenvalues = np.ones((dimension,), dtype=np.float_)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=np.float_)

    def update_eigensystem(self):
        for i in range(len(self.C)):
            for j in range(i):
                self.C[i,j] = self.C[j,i]
        
        self.eigenvalues, self.eigenbasis = eig(self.C) 
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenbasis = np.real(self.eigenbasis)
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)

        for i in range(len(self.C)):
            for j in range(i+1):
                self.invsqrt[i,j] = self.invsqrt[j,i] = sum(
                        self.eigenbasis[i,k] * self.eigenbasis[j,k]
                        / self.eigenvalues[k] ** 0.5 for k in range(len(self.C))
                    )

class CMA_ES_Algorithm:

    def __init__(self, num_to_evaluate,mutation_power,feature_map):
        self.population_size=parsed_toml["PopulationSize"]
        self.num_parents = self.population_size // 2
        self.feature_map=feature_map
        self.allRecords=pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])
        self.mutation_power = mutation_power
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0

        self.mean = np.asarray([0.0] * num_params)
        self.population = []
        
        self.successful_individuals=[]

        # Setup recombination weights
        self.weights = [math.log(self.num_parents + 0.5) \
                - math.log(i+1) for i in range(self.num_parents)] 
        total_weights = sum(self.weights)
        self.weights = np.array([w/total_weights for w in self.weights])
        self.mueff = sum(self.weights) ** 2 / sum(self.weights ** 2)
	
        # Setup strategy parameters for adaptation
        self.cc = (4+self.mueff/num_params)/(num_params+4 + 2*self.mueff/num_params)
        self.cs = (self.mueff+2)/(num_params+self.mueff+5)
        self.c1 = 2/((num_params+1.3)**2+self.mueff)
        self.cmu = min(1-self.c1,2*(self.mueff-2+1/self.mueff)/((num_params+2)**2+self.mueff))
        self.damps = 1 + 2*max(0,math.sqrt((self.mueff-1)/(num_params+1))-1)+self.cs
        self.chiN = num_params**0.5 * (1-1/(4*num_params)+1./(21*num_params**2))

        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        unscaled_params = \
            [self.mutation_power * eigenval ** 0.5 * gaussian() \
                for eigenval in self.C.eigenvalues]
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        ind = Individual()
        ind.param_vector = unscaled_params
        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector)
        ind.level=level
        #ind.features = (num_non_empty, num_enemies)
        return ind
    
    def return_evaluated_individual(self, ind):
        ind.make_features()
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.allRecords.loc[ind.ID]=["CMA-ES"]+[ind.param_vector]+ind.statsList+[ind.features[0]]+[ind.features[1]]

        RecordFrequency=parsed_toml["RecordFrequency"]
        if self.individuals_evaluated % RecordFrequency == 0:
            """
            #show the map of success levels
            image_path = os.path.join(success_map, 'gen_{0:03d}.png'.format(self.individuals_evaluated // RecordFrequency))
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                pts = make_record_frame([x.features+(x.fitness,) for x in elites])
                #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
                create_image(pts, image_path)

            """

            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open(records+"\\EliteLog.csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID,["emitterName",'completionPercentage',"behavior feature X","behavior feature Y"]].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()

        self.population.append(ind)
        if len(self.population) < self.population_size:
            return

        for cur in self.population:
            self.feature_map.add(cur)

        # Sort by fitness
        parents = sorted(self.population, key=lambda x: x.fitness)[::-1]
        parents = parents[:self.num_parents]
        #print('----', parents[0].fitness)

        # Recombination of the new mean
        old_mean = self.mean
        self.mean = sum(ind.param_vector * w for ind, w in zip(parents, self.weights))
        #print('mean', self.mean)

        # Update the evolution path
        y = self.mean - old_mean
        z = np.matmul(self.C.invsqrt, y)
        self.ps = (1-self.cs) * self.ps +\
           (math.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.mutation_power) * z
        left = sum(x**2 for x in self.ps) / num_params \
             / (1-(1-self.cs)**(2*self.individuals_evaluated / self.population_size)) 
        right = 2 + 4./(num_params+1)
        hsig = 1 if left < right else 0

        self.pc = (1-self.cc) * self.pc + \
            hsig * math.sqrt(self.cc*(2-self.cc)*self.mueff) * y

        # Adapt the covariance matrix
        c1a = self.c1 * (1 - (1-hsig**2) * self.cc * (2 - self.cc))
        self.C.C *= (1 - c1a - self.cmu)
        self.C.C += self.c1 * np.outer(self.pc, self.pc)
        for k, w in enumerate(self.weights):
            dv = parents[k].param_vector - old_mean
            self.C.C += w * self.cmu * np.outer(dv, dv) / (self.mutation_power ** 2)
        #print('C', np.amin(self.C.C))

        # Updated the covariance matrix decomposition and inverse
        self.C.update_eigensystem()
	
        # Update sigma
        cn, sum_square_ps = self.cs / self.damps, sum(x**2 for x in self.ps)
        self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        """
        # Visualize current generation
        for cur in self.population:
            if(cur.fitness==1.0):
                self.successful_individuals.append(cur)
        image_path = os.path.join('cmaes_map', 'gen_{0:03d}.png'.format(self.individuals_evaluated // self.population_size))
        #print(image_path)
        pts = make_record_frame([x.features+(x.fitness,) for x in self.successful_individuals])
        create_image(pts, image_path)
        """

        # Reset the population
        self.population.clear()

def run_cma_es(num_to_evaluate, mutation_power):

    resolution = (parsed_toml["CMAMESetting"]["StartResolution"],parsed_toml["CMAMESetting"]["EndResolution"])
    #feature_ranges = [(-(num_params*boundary_value//2), num_params*boundary_value//2)] * 2
    feature_ranges=[(parsed_toml["CMAMESetting"]["X_Low"],parsed_toml["CMAMESetting"]["X_High"]),(parsed_toml["CMAMESetting"]["Y_Low"],parsed_toml["CMAMESetting"]["Y_High"])]

    feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolution)
    cmaes = CMA_ES_Algorithm(num_to_evaluate,mutation_power,feature_map)

    while cmaes.is_running():
        ind = cmaes.generate_individual()
        ind.fitness = evaluate(ind)
        cmaes.return_evaluated_individual(ind)
    
    #output all records to csv files
    cmaes.allRecords.to_csv("records\\AllRecords.csv")

class FeatureMap:

   def __init__(self, max_individuals, feature_ranges, resolutions):
      self.max_individuals = max_individuals
      self.feature_ranges = feature_ranges
      self.resolutions = resolutions

      self.elite_map = {}
      self.elite_indices = []
      self.num_groups = -1

      self.num_individuals_added = 0

   def clone_blank(self):
      copy_map = FeatureMap(self.max_individuals, self.feature_ranges, self.resolutions)
      copy_map.num_individuals_added = self.num_individuals_added
      copy_map.num_groups = self.num_groups
      return copy_map

   def get_size(self, portion_done):
      rval = self.resolutions[1] - self.resolutions[0]
      size = int(((portion_done+1e-9) * rval) + self.resolutions[0])
      return min(size, self.resolutions[1])

   def get_feature_index(self, feature_id, feature):
      feature_range = self.feature_ranges[feature_id]
      if feature <= feature_range[0]:
         return 0
      if feature >= feature_range[1]:
         return self.num_groups-1

      gap = feature_range[1] - feature_range[0] + 1
      pos = feature - feature_range[0]
      index = int(self.num_groups * pos / gap)
      return index

   def get_index(self, cur):
      return tuple(self.get_feature_index(i, f) for i, f in enumerate(cur.features))

   def add_to_map(self, to_add):
      index = self.get_index(to_add)

      replaced_elite = False
      if index not in self.elite_map:
         self.elite_indices.append(index)
         self.elite_map[index] = to_add
         replaced_elite = True
         to_add.delta = (1, to_add.fitness)
      elif self.elite_map[index].fitness < to_add.fitness:
         to_add.delta = (0, to_add.fitness-self.elite_map[index].fitness)
         self.elite_map[index] = to_add
         replaced_elite = True

      return replaced_elite

   def remap(self, next_num_groups):
      self.num_groups = next_num_groups

      all_elites = []
      for index in self.elite_map:
         all_elites.append(self.elite_map[index])

      self.elite_indices = []
      self.elite_map = {}
      for cur in all_elites:
         self.add_to_map(cur)

   def add(self, to_add):
      self.num_individuals_added += 1
      portion_done = 1.0 * self.num_individuals_added / self.max_individuals
      next_size = self.get_size(portion_done)
      if next_size != self.num_groups:
         self.remap(next_size)

      replaced_elite = self.add_to_map(to_add)
      return replaced_elite

   def get_random_elite(self):
      pos = random.randint(0, len(self.elite_indices)-1)
      index = self.elite_indices[pos]
      return self.elite_map[index]


class ImprovementEmitter:

    def __init__(self, mutation_power, feature_map):
        #self.population_size = int(4.0+math.floor(3.0*math.log(num_params))) * 2
        self.population_size=parsed_toml["PopulationSize"]
        #print('pop size', self.population_size)
        self.sigma = mutation_power
        self.num_released = 0
        
        self.population = []
        self.feature_map = feature_map
        self.num_features = len(self.feature_map.feature_ranges)
        #print('num_features', self.num_features)
        
        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector
        
        #print('RESET --------------')
        #print('new mean:', self.mean)
 
        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True

        return False

    def generate_individual(self):
        unscaled_params = \
            [self.mutation_power * eigenval ** 0.5 * gaussian() \
                for eigenval in self.C.eigenvalues]
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        ind = Individual()
        ind.param_vector = unscaled_params
        #print(ind.param_vector)
        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector)
        ind.level=level
        #print(level)
        #sys.stdout.flush()
        ind.emitter_name="ImprovementEmitter"

        #half = len(ind.param_vector) // 2
        #capped_vals = [ind.reduce(x) for x in ind.param_vector]
        #ind.features = (num_non_empty, num_enemies)
        self.num_released += 1
        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        if len(self.population) < self.population_size:
            print("return")
            return

        # Only filter by this generation
        parents = []
        for cur in self.population:
            if self.feature_map.add(cur):
                parents.append(cur)
        num_parents = len(parents)
        needs_restart = num_parents == 0


        #print('parents:', num_parents)

        # Only update if there are parents
        if num_parents > 0:
            print("did improve")
            #sys.stdout.flush()
            parents = sorted(parents, key=lambda x: x.delta)[::-1]

            # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                    - math.log(i+1) for i in range(num_parents)] 
            total_weights = sum(weights)
            weights = np.array([w/total_weights for w in weights])
        
            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4+mueff/num_params)/(num_params+4 + 2*mueff/num_params)
            cs = (mueff+2)/(num_params+mueff+5)
            c1 = 2/((num_params+1.3)**2+mueff)
            cmu = min(1-c1,2*(mueff-2+1/mueff)/((num_params+2)**2+mueff))
            damps = 1 + 2*max(0,math.sqrt((mueff-1)/(num_params+1))-1)+cs
            chiN = num_params**0.5 * (1-1/(4*num_params)+1./(21*num_params**2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1-cs) * self.ps +\
                (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x**2 for x in self.ps) / num_params
            right = 2 + 4./(num_params+1)
            hsig = 1 if left < right else 0

            self.pc = (1-cc) * self.pc + \
                hsig * math.sqrt(cc*(2-cc)*mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1-hsig**2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()
	
            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x**2 for x in self.ps)
            self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        else:
            print("no improve")
            #sys.stdout.flush()
        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()


class RandomDirectionEmitter:

    def __init__(self, mutation_power, feature_map):
        #self.population_size = int(4.0+math.floor(3.0*math.log(num_params)))
        self.population_size=parsed_toml["PopulationSize"]
        #print('pop size', self.population_size)
        self.sigma = mutation_power
        self.num_released = 0
        
        self.population = []
        self.feature_map = feature_map
        self.num_features = len(self.feature_map.feature_ranges)
        #print('num_features', self.num_features)
        
        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector
        self.direction = np.asarray([gaussian() for _ in range(self.num_features)])
        
        #print('RESET --------------')
        #print('new mean:', self.mean, 'direction:', self.direction)
 
        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True

        return False

    def generate_individual(self):
        unscaled_params = \
            [self.mutation_power * eigenval ** 0.5 * gaussian() \
                for eigenval in self.C.eigenvalues]
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        ind = Individual()
        ind.param_vector = unscaled_params
        self.num_released += 1
        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector)
        ind.level=level
        #print(level)
        #sys.stdout.flush()
        ind.emitter_name="RandomDirectionEmitter"

        #half = len(ind.param_vector) // 2
        #capped_vals = [ind.reduce(x) for x in ind.param_vector]
        #ind.features = (num_non_empty, num_enemies)
        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        if len(self.population) < self.population_size:
            print("return")
            return

        # Only filter by this generation
        did_improve = False
        num_improve = 0
        local_map = self.feature_map.clone_blank()
        parents = []
        for cur in self.population:
            local_map.add(cur)
            if self.feature_map.add(cur):
                did_improve = True
                num_improve += 1
        for index in local_map.elite_indices:
            parents.append(local_map.elite_map[index])
        num_parents = len(parents)

        # Calculate the behavior mean
        feature_mean = sum([np.array(ind.features) for ind in self.population]) / self.population_size
        #print('emitter', ind.emitter_id)
        #print('feature mean', feature_mean)
        #print('parents:', num_parents, num_improve)
        needs_restart = not did_improve

        # Only update if there are parents
        if did_improve:
            print("did improve")
            #sys.stdout.flush()
            for ind in parents:
                dv = np.asarray(ind.features) - feature_mean
                ind.projection = np.dot(self.direction, dv)
            parents = sorted(parents, key=lambda x: x.projection)

            # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                    - math.log(i+1) for i in range(num_parents)] 
            total_weights = sum(weights)
            weights = np.array([w/total_weights for w in weights])
        
            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4+mueff/num_params)/(num_params+4 + 2*mueff/num_params)
            cs = (mueff+2)/(num_params+mueff+5)
            c1 = 2/((num_params+1.3)**2+mueff)
            cmu = min(1-c1,2*(mueff-2+1/mueff)/((num_params+2)**2+mueff))
            damps = 1 + 2*max(0,math.sqrt((mueff-1)/(num_params+1))-1)+cs
            chiN = num_params**0.5 * (1-1/(4*num_params)+1./(21*num_params**2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1-cs) * self.ps +\
                (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x**2 for x in self.ps) / num_params
            right = 2 + 4./(num_params+1)
            hsig = 1 if left < right else 0

            self.pc = (1-cc) * self.pc + \
                hsig * math.sqrt(cc*(2-cc)*mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1-hsig**2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()
	
            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x**2 for x in self.ps)
            self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        else:
            print("no improve")
            #sys.stdout.flush()

        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()

class OptimizingEmitter:

    def __init__(self, mutation_power, feature_map):
        #self.population_size = int(4.0+math.floor(3.0*math.log(num_params))) * 2
        self.population_size=parsed_toml["PopulationSize"]
        #print('pop size', self.population_size)
        self.sigma = mutation_power
        self.num_released = 0
        
        self.population = []
        self.feature_map = feature_map
        self.num_features = len(self.feature_map.feature_ranges)
        #print('num_features', self.num_features)
        
        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector
        
        #print('RESET --------------')
        #print('new mean:', self.mean)
 
        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True
        if abs(parents[0].fitness-parents[-1].fitness) < 1e-6:
            return True

        return False

    def generate_individual(self):
        unscaled_params = \
            [self.mutation_power * eigenval ** 0.5 * gaussian() \
                for eigenval in self.C.eigenvalues]
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        ind = Individual()
        ind.param_vector = unscaled_params
        ind.emitter_name="OptimizingEmitter"
        self.num_released += 1

        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector)
        ind.level=level
        #print(level)
        #sys.stdout.flush()

        #half = len(ind.param_vector) // 2
        #capped_vals = [ind.reduce(x) for x in ind.param_vector]
        #ind.features = (num_non_empty, num_enemies)

        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        if len(self.population) < self.population_size:
            print("return")
            return

        # Continue optimizing if we are improving the map.
        did_improve = True
        for cur in self.population:    
            self.feature_map.add(cur)
        needs_restart = not did_improve

        # Only update if there are parents
        if did_improve:
            print("did improve")
            #sys.stdout.flush()

            num_parents = self.population_size // 2
            parents = sorted(self.population, key=lambda x: x.fitness)[::-1]
            parents = parents[:num_parents]
            #print('----', parents[0].fitness)

                # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                    - math.log(i+1) for i in range(num_parents)] 
            total_weights = sum(weights)
            weights = np.array([w/total_weights for w in weights])
        
            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4+mueff/num_params)/(num_params+4 + 2*mueff/num_params)
            cs = (mueff+2)/(num_params+mueff+5)
            c1 = 2/((num_params+1.3)**2+mueff)
            cmu = min(1-c1,2*(mueff-2+1/mueff)/((num_params+2)**2+mueff))
            damps = 1 + 2*max(0,math.sqrt((mueff-1)/(num_params+1))-1)+cs
            chiN = num_params**0.5 * (1-1/(4*num_params)+1./(21*num_params**2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1-cs) * self.ps +\
            (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x**2 for x in self.ps) / num_params
            right = 2 + 4./(num_params+1)
            hsig = 1 if left < right else 0


            self.pc = (1-cc) * self.pc + \
                hsig * math.sqrt(cc*(2-cc)*mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1-hsig**2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()
	
            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x**2 for x in self.ps)
            self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        else:
            print("no improve")
            #sys.stdout.flush()

        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()


class CMA_ME_Algorithm:

    def __init__(self, mutation_power, num_to_evaluate, feature_map):
        self.emitters = []
        self.records=[]
        self.allRecords=pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])
        if parsed_toml["EmitterSelection"]["RandomDirectionEmitter"]:
            self.emitters += [RandomDirectionEmitter(mutation_power, feature_map) for i in range(1)]
            self.records+=[pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])]
        
        if parsed_toml["EmitterSelection"]["ImprovementEmitter"]:
            self.emitters += [ImprovementEmitter(mutation_power, feature_map) for i in range(1)]
            self.records+=[pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])]
        
        if parsed_toml["EmitterSelection"]["OptimizingEmitter"]:
            self.emitters += [OptimizingEmitter(mutation_power, feature_map) for i in range(1)]
            self.records+=[pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])]
        
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0
        self.feature_map = feature_map
       

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        
        pos = 0
        emitter = self.emitters[0]
        for i in range(1, len(self.emitters)):
            if self.emitters[i].num_released < emitter.num_released:
                emitter = self.emitters[i]
                pos = i
        
        ind = emitter.generate_individual()
        ind.emitter_id = pos
        return ind

    def return_evaluated_individual(self, ind):
        ind.make_features() #make behavior features, now it does nothing
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.emitters[ind.emitter_id].return_evaluated_individual(ind)
        self.records[ind.emitter_id].loc[ind.ID]=[ind.emitter_name]+[ind.param_vector]+ind.statsList+[ind.features[0]]+[ind.features[1]]
        self.allRecords.loc[ind.ID]=[ind.emitter_name]+[ind.param_vector]+ind.statsList+[ind.features[0]]+[ind.features[1]]
        
        
        # Visualize this generation
        RecordFrequency=parsed_toml["RecordFrequency"]
        if self.individuals_evaluated % RecordFrequency == 0:
            """
            #show the map of success levels
            image_path = os.path.join(success_map, 'gen_{0:03d}.png'.format(self.individuals_evaluated // RecordFrequency))
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                pts = make_record_frame([x.features+(x.fitness,) for x in elites])
                #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
                create_image(pts, image_path)

            """

            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open(records+"\\EliteLog.csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID,["emitterName","completionPercentage","behavior feature X","behavior feature Y"]].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()

               

def run_cma_me(num_to_evaluate, mutation_power=parsed_toml["CMAMESetting"]["MutationPower"]):
    resolution = (parsed_toml["CMAMESetting"]["StartResolution"],parsed_toml["CMAMESetting"]["EndResolution"])
    #feature_ranges = [(-(num_params*boundary_value//2), num_params*boundary_value//2)] * 2
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
    for i in range (0,len(cma_me.records)):
        cma_me.records[i].to_csv(records+"\\Emitter"+str(i)+".csv")
    cma_me.allRecords.to_csv(records+"\\AllRecords.csv")

    """
    #to heatmap
    latentMap = [0] * (feature_ranges[1][1]+1)
    for i in range(feature_ranges[1][1]+1):
        latentMap[i] = [0] * (feature_ranges[0][1]+1)

    elites = [cma_me.feature_map.elite_map[x] for x in cma_me.feature_map.elite_map]
    for elite in elites:
        if elite.features[0]<=feature_ranges[0][1] and elite.features[1]<=feature_ranges[1][1]:
            latentMap[elite.features[1]][elite.features[0]]=elite.fitness
        else:
            elites.remove(elite)
    
    image_path = os.path.join(success_map, str(len(elites))+'cells.png')
    if(len(elites)!=0):
        #pts = make_record_frame([x.features+(x.fitness,) for x in elites])
        #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
        df=pd.DataFrame(latentMap)
        with sns.axes_style("white"):
            plt.figure(figsize=(20,20))
            #pts=pts.pivot("y", "x","f")
            #g = sns.heatmap(pts)
            g=sns.heatmap(df,cmap=sns.color_palette("Blues"),robust=True)
            g.set(xlabel='Number of Non-Empty Block', ylabel='Number of Enemies')
            fig=g.get_figure()
            fig.savefig(image_path)
        plt.close('all')
    """

class MapElitesAlgorithm:

    def __init__(self, mutation_power, initial_population, num_to_evaluate, feature_map):
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power = mutation_power
        self.allRecords=pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])
        
    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        
        ind = Individual()
        if self.individuals_evaluated < self.initial_population:
            unscaled_params = \
                [np.random.uniform(low=-boundary_value, high=boundary_value) for _ in range(num_params)]
            ind.param_vector = unscaled_params
        else:
            parent = self.feature_map.get_random_elite()
            unscaled_params = \
                [parent.param_vector[i] + self.mutation_power * gaussian() for i in range(num_params)]
            ind.param_vector = unscaled_params

        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector)
        ind.level=level
        #print(level)
        #sys.stdout.flush()

        #half = len(ind.param_vector) // 2
        #capped_vals = [ind.reduce(x) for x in ind.param_vector]
        #ind.features = (num_non_empty, num_enemies)
        return ind

    def return_evaluated_individual(self, ind):

        ind.make_features() #now this does nothing
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        self.feature_map.add(ind)
        self.allRecords.loc[ind.ID]=["MAP-Elite"]+[ind.param_vector]+ind.statsList+[ind.features[0]]+[ind.features[1]]

        print("Evaluated One Individual")
        #sys.stdout.flush()

        
        # Visualize this generation
        RecordFrequency=parsed_toml["RecordFrequency"]
        if self.individuals_evaluated % RecordFrequency == 0:
            """
            #show the map of success levels
            image_path = os.path.join(success_map, 'gen_{0:03d}.png'.format(self.individuals_evaluated // RecordFrequency))
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                pts = make_record_frame([x.features+(x.fitness,) for x in elites])
                #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
                create_image(pts, image_path)
            """
        

            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("records\\EliteLog.csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID,["emitterName",'completionPercentage',"behavior feature X","behavior feature Y"]].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()



def run_map_elites(num_to_evaluate, initial_population, mutation_power=parsed_toml["MAPEliteSetting"]["MutationPower"]):
    resolution = (parsed_toml["MAPEliteSetting"]["StartResolution"],parsed_toml["MAPEliteSetting"]["EndResolution"])
    #feature_ranges = [(-(num_params*boundary_value//2), num_params*boundary_value//2)] * 2
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
    me.allRecords.to_csv("records\\AllRecords.csv")

    """
    #To HeatMap
    latentMap = [0] * (feature_ranges[1][1]+1)
    for i in range(feature_ranges[1][1]+1):
        latentMap[i] = [0] * (feature_ranges[0][1]+1)

    elites = [me.feature_map.elite_map[x] for x in me.feature_map.elite_map]
    for elite in elites:
        if elite.features[0]<=feature_ranges[0][1] and elite.features[1]<=feature_ranges[1][1]:
            latentMap[elite.features[1]][elite.features[0]]=elite.fitness
        else:
            elites.remove(elite)
    
    image_path = os.path.join(success_map, str(len(elites))+'cells.png')
    if(len(elites)!=0):
        #pts = make_record_frame([x.features+(x.fitness,) for x in elites])
        #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
        df=pd.DataFrame(latentMap)
        with sns.axes_style("white"):
            plt.figure(figsize=(20,20))
            #pts=pts.pivot("y", "x","f")
            #g = sns.heatmap(pts)
            g=sns.heatmap(df,cmap=sns.color_palette("Blues"),robust=True)
            g.set(xlabel='Number of Non-Empty Block', ylabel='Number of Enemies')
            fig=g.get_figure()
            fig.savefig(image_path)
        plt.close('all')
    """


class ISOLineDDAlgorithm:

    def __init__(self, mutation_power1,mutation_power2, initial_population, num_to_evaluate, feature_map):
        self.num_to_evaluate = num_to_evaluate
        self.initial_population = initial_population
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power1 = mutation_power1
        self.mutation_power2=mutation_power2
        self.allRecords=pd.DataFrame(columns=['emitterName','latentVector', 'completionPercentage','jumpActionsPerformed','killsTotal','livesLeft','coinsCollected','remainingTime (20-timeSpent)','behavior feature X','behavior feature Y'])
        
    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        
        ind = Individual()
        if self.individuals_evaluated < self.initial_population:
            unscaled_params = \
                [np.random.uniform(low=-boundary_value, high=boundary_value) for _ in range(num_params)]
            ind.param_vector = unscaled_params
        else:
            pos = random.randint(0, len(self.feature_map.elite_indices)-1)
            index1 = self.feature_map.elite_indices[pos]
            pos = random.randint(0, len(self.feature_map.elite_indices)-1)
            index2 = self.feature_map.elite_indices[pos]
            
            parent1=self.feature_map.elite_map[index1]
            parent2=self.feature_map.elite_map[index2]

            unscaled_params = \
                [parent1.param_vector[i] + self.mutation_power1 * gaussian() + self.mutation_power2 * (parent1.param_vector[i]-parent2.param_vector[i]) * gaussian() for i in range(num_params)]
            ind.param_vector = unscaled_params

        #level,num_non_empty,num_enemies=gan_generate(ind.param_vector)
        level=gan_generate(ind.param_vector)
        ind.level=level
        #print(level)
        #sys.stdout.flush()

        #half = len(ind.param_vector) // 2
        #capped_vals = [ind.reduce(x) for x in ind.param_vector]
        #ind.features = (num_non_empty, num_enemies)
        return ind

    def return_evaluated_individual(self, ind):

        ind.make_features() #now this does nothing
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        
        self.feature_map.add(ind)
        self.allRecords.loc[ind.ID]=["MAP-Elite"]+[ind.param_vector]+ind.statsList+[ind.features[0]]+[ind.features[1]]

        print("Evaluated One Individual")
        #sys.stdout.flush()

        
         # Visualize this generation
        RecordFrequency=parsed_toml["RecordFrequency"]
        if self.individuals_evaluated % RecordFrequency == 0:
            """
            #show the map of success levels
            image_path = os.path.join(success_map, 'gen_{0:03d}.png'.format(self.individuals_evaluated // RecordFrequency))
            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                pts = make_record_frame([x.features+(x.fitness,) for x in elites])
                #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
                create_image(pts, image_path)
            """
        

            elites = [self.feature_map.elite_map[x] for x in self.feature_map.elite_map]
            if(len(elites)!=0):
                logFile=open("records\\EliteLog.csv","a")
                rowData=[]
                for x in elites:
                    currElite=[x.ID]
                    currElite+=self.allRecords.loc[x.ID,["emitterName",'completionPercentage',"behavior feature X","behavior feature Y"]].tolist()
                    rowData.append(currElite)
                wr = csv.writer(logFile, dialect='excel')
                wr.writerow(rowData)
                logFile.close()



def run_ISOLineDD(num_to_evaluate, initial_population, mutation_power1,mutation_power2):
    resolution = (parsed_toml["MAPEliteSetting"]["StartResolution"],parsed_toml["MAPEliteSetting"]["EndResolution"])
    #feature_ranges = [(-(num_params*boundary_value//2), num_params*boundary_value//2)] * 2
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
    me.allRecords.to_csv("records\\AllRecords.csv")


    """
    #To HeatMap
    latentMap = [0] * (feature_ranges[1][1]+1)
    for i in range(feature_ranges[1][1]+1):
        latentMap[i] = [0] * (feature_ranges[0][1]+1)

    elites = [me.feature_map.elite_map[x] for x in me.feature_map.elite_map]
    for elite in elites:
        if elite.features[0]<=feature_ranges[0][1] and elite.features[1]<=feature_ranges[1][1]:
            latentMap[elite.features[1]][elite.features[0]]=elite.fitness
        else:
            elites.remove(elite)
    
    image_path = os.path.join(success_map, str(len(elites))+'cells.png')
    if(len(elites)!=0):
        #pts = make_record_frame([x.features+(x.fitness,) for x in elites])
        #print(image_path, len(elites) / (self.feature_map.resolutions[-1] ** 2), total_fitness)
        df=pd.DataFrame(latentMap)
        with sns.axes_style("white"):
            plt.figure(figsize=(20,20))
            #pts=pts.pivot("y", "x","f")
            #g = sns.heatmap(pts)
            g=sns.heatmap(df,cmap=sns.color_palette("Blues"),robust=True)
            g.set(xlabel='Number of Non-Empty Block', ylabel='Number of Enemies')
            fig=g.get_figure()
            fig.savefig(image_path)
        plt.close('all')
    """
    
    #print(best)

if __name__ == '__main__':
    print("READY") # Java loops until it sees this special signal
    #sys.stdout.flush() # Make sure Java can sense this output before Python blocks waiting for input 
    NumSimulations=parsed_toml["NumSimulations"]
    AlgorithmToRun=parsed_toml["Algorithm"]
    print(AlgorithmToRun)
    if(AlgorithmToRun=="CMAES"):
        run_cma_es(NumSimulations,mutation_power=parsed_toml["CMAMESetting"]["MutationPower"])
    if(AlgorithmToRun=="CMAME"):
        run_cma_me(NumSimulations, mutation_power=parsed_toml["CMAMESetting"]["MutationPower"])
    if(AlgorithmToRun=="MAPELITE"):
        run_map_elites(NumSimulations,initial_population=parsed_toml["MAPEliteSetting"]["InitialPopulation"],mutation_power=parsed_toml["MAPEliteSetting"]["MutationPower"])
    if(AlgorithmToRun=="ISOLineDD"):
        run_ISOLineDD(NumSimulations,initial_population=parsed_toml["ISOLineDDSetting"]["InitialPopulation"],mutation_power1=parsed_toml["ISOLineDDSetting"]["MutationPower1"],mutation_power2=parsed_toml["ISOLineDDSetting"]["MutationPower2"])
    print("saved")
    #sys.stdout.flush()
    #run_map_elites(50000, 100, mutation_power=0.3)
