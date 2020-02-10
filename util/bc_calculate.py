import numpy as np
import math
import os
import json
import toml

parsed_toml=toml.load("Searching/config/cma_me.tml")

COIN=parsed_toml["GameSetting"]["COIN"]
GROUND = parsed_toml["GameSetting"]["GROUND"]
ENEMY = parsed_toml["GameSetting"]["ENEMY"]
PIPE = parsed_toml["GameSetting"]["PIPE"]
EMPTY = parsed_toml["GameSetting"]["EMPTY"]

Ground=parsed_toml["HeightSeparator"]["Ground"]
AboveGround=parsed_toml["HeightSeparator"]["AboveGround"]
MiddleLevel=parsed_toml["HeightSeparator"]["MiddleLevel"]
HigherLevel=parsed_toml["HeightSeparator"]["HigherLevel"]

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