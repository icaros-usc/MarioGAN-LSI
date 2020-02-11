import random
import math
import pandas as pd
import numpy as np
import toml
from numpy.linalg import eig
import json


parsed_toml=toml.load("Searching/config/cma_me.tml")
num_params = parsed_toml["ModelParameter"]["num_params"]
boundary_value = parsed_toml["ModelParameter"]["boundary_value"]
batchSize = parsed_toml["ModelParameter"]["batchSize"]
nz = parsed_toml["ModelParameter"]["nz"]
success_map=parsed_toml["LogPaths"]["success_map"]
records=parsed_toml["LogPaths"]["recordsCSV"]

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

def gaussian():
    u1 = 1.0 - random.random()
    u2 = 1.0 - random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)


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

class FeatureMap:

   def __init__(self, max_individuals, feature_ranges, resolutions=(60,60)):
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
