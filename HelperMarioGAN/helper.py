import numpy as np
import math
import os

def calc_tp_count(lvl, p_size, border_value=-1):
    padded_lvl = np.pad(lvl, p_size, constant_values=border_value)
    pattern_dict = {}
    for (y, x), v in np.ndenumerate(lvl):
        sy, sx = y+p_size-p_size//2, x+p_size-p_size//2
        pattern = str(padded_lvl[sy:sy+p_size,sx:sx+p_size])
        if not (pattern in pattern_dict):
            pattern_dict[pattern] = 0
        pattern_dict[pattern] += 1
    return pattern_dict

def calc_tp_kldiv(test_count, train_count, epsilon = 1e-6):
    p_dict = test_count
    q_dict = train_count
    t_patterns = set()
    t_patterns.update(p_dict.keys())
    t_patterns.update(q_dict.keys())
    total_p = sum(p_dict.values())
    total_q = sum(q_dict.values())

    mp_dict = {}
    mq_dict = {}
    mp_total = 0
    mq_total = 0
    for x in t_patterns:
        p_dash = epsilon/((total_p + epsilon) * (1 + epsilon))
        q_dash = epsilon/((total_q + epsilon) * (1 + epsilon))
        if x in p_dict:
            p_dash = (p_dict[x] + epsilon) / ((total_p + epsilon) * (1 + epsilon))
        if x in q_dict:
            q_dash = (q_dict[x] + epsilon) / ((total_p + epsilon) * (1 + epsilon))
        mp_dict[x] = p_dash
        mq_dict[x] = q_dash
        mp_total += p_dash
        mq_total += q_dash

    value = 0
    for x in t_patterns:
        p_dash = mp_dict[x] / mp_total
        q_dash = mq_dict[x] / mq_total
        value += p_dash * math.log(p_dash/q_dash)
    return value

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
