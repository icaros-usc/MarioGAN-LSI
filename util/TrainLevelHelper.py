import numpy as np
import math
import os
import random
import pandas as pd



def get_lvls(folder):
    lvls = []
    files = sorted([f for f in os.listdir(folder) if 'lvl' in f])
    for f in files:
        f=open(os.path.join(folder, f))
        lvl=f.readlines()
        clean_level = []
        for l in lvl:
            if len(l.strip()) > 0:
                clean_level.append(l.strip())
        lvls.append(clean_level)
    return lvls

def build_indeces(lvls, compressed=True):
    normal_sames = ['*bB', 'gE', '!Q', '@?']
    compressed_sames = ['-12', '*bB', 'yY', 'gGE', 'kK', 'rR', 'tT', 'SCUL', '!Q', '@?']
    same = normal_sames
    if compressed:
        same = compressed_sames

    full_string = ''
    for lvl in lvls:
        full_string += ''.join(lvl)
    uniqueSymbols = set(full_string)
    str2index = {}
    index2str = {}
    c_index = 0
    if same is not None:
        for g in same:
            for c in g:
                str2index[c] = c_index
            index2str[c_index] = g[0]
            c_index += 1
    for symbol in uniqueSymbols:
        if symbol not in str2index:
            str2index[symbol] = c_index
            index2str[c_index] = symbol
            c_index += 1
    return str2index, index2str, c_index

def get_integer_lvl(lvl, str2index):
    result = []
    numpyLvl = np.zeros((len(lvl), len(lvl[0])))
    for y in range(len(lvl)):
        for x in range(len(lvl[y])):
            c = lvl[y][x]
            numpyLvl[y][x] = str2index[c]
    return numpyLvl.astype(np.uint8)

def get_string_lvl(array, index2str):
    result = ""
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            result += index2str[array[y][x]]
        result += "\n"
    return result

def get_windows(lvls, width, indeces):
    result = []
    for lvl in lvls:
        for shift in range(len(lvl[0]) - width + 1):
            window = np.zeros((len(lvl), width))
            for y in range(len(lvl)):
                for x in range(width):
                    c = lvl[y][x+shift]
                    window[y][x] = indeces[c]
            result.append(window)
    return np.array(result).astype(np.uint8)

def get_windows_from_folder(folder, width=56, compressed=True):
    lvls = get_lvls(folder)
    str2index, index2str, types = build_indeces(lvls, compressed)
    windows = get_windows(lvls, width, str2index)
    return windows, types, index2str
