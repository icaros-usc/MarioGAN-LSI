import numpy as np
import math
import os
import json
import toml
from util.SearchHelper import *


def calc_higher_level_non_empty_blocks(ind,result):
    HigherLevel=9
    EMPTY=0
    im=np.array(json.loads(ind.level))
    higher=im[0:HigherLevel,:]
    num_non_empty=len(higher[higher!=EMPTY])
    #num_non_empty=len(im[im!=EMPTY])
    return num_non_empty

def calc_num_enemies(ind,result):
    ENEMY=[2,3,4,5]
    im=np.array(json.loads(ind.level))
    num_enemies =  len (im[np.isin(im,ENEMY)])
    return num_enemies


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

#get the count for level 1 [:,:56]
f=open("../Mario-AI-Framework/levels/original/lvl-1.txt")
lvl=f.readlines()
level = []
for l in lvl:
    row=[]
    if len(l.strip()) > 0:
        for i in l.strip():
            row.append(i)
        level.append(row)
level=np.array(level)[:,:56]
count1=calc_tp_count(level,1)

#get the count for level 3 [:,:56]
f=open("../Mario-AI-Framework/levels/original/lvl-3.txt")
lvl=f.readlines()
level = []
for l in lvl:
    row=[]
    if len(l.strip()) > 0:
        for i in l.strip():
            row.append(i)
        level.append(row)
level=np.array(level)[:,:56]
count3=calc_tp_count(level,1)

def calc_kldivergent_level1(ind,result):
    strlevel=[]
    number_level=eval(ind.level)
    for x in number_level:
        row=[]
        for y in x:
            row.append(get_char(y))
        strlevel.append(row)
    strlevel=np.array(strlevel)[:,:56]
    ind_count=calc_tp_count(strlevel,1)
    return calc_tp_kldiv(ind_count,count1)

def calc_kldivergent_level3(ind,result):
    strlevel=[]
    number_level=eval(ind.level)
    for x in number_level:
        row=[]
        for y in x:
            row.append(get_char(y))
        strlevel.append(row)
    strlevel=np.array(strlevel)[:,:56]
    ind_count=calc_tp_count(strlevel,1)
    return calc_tp_kldiv(ind_count,count3)

def calc_jump(ind,result):
    num_jumps=result.getNumJumps()
    if num_jumps>0:
        return 1
    else:
        return 0

def calc_high_jump(ind,result):
    max_jump_frame=result.getMaxJumpAirTime()
    if max_jump_frame >12:
        return 1
    else:
        return 0

def calc_long_jump(ind,result):
    max_X_jump=result.getMaxXJump()
    if max_X_jump>120:
        return 1
    else:
        return 0

def calc_stomp(ind,result):
    num_stomp=result.getKillsByStomp()
    if(num_stomp>0):
        return 1
    else:
        return 0

def calc_shell_kill(ind,result):
    num_shell_kill=result.getKillsByShell()
    if num_shell_kill>0:
        return 1
    else:
        return 0

def calc_fall_kill(ind,result):
    num_fall_kill=result.getKillsByFall()
    if num_fall_kill>0:
        return 1
    else:
        return 0

def calc_mushroom(ind,result):
    num_mushroom=result.getNumCollectedMushrooms()
    if(num_mushroom>0):
        return 1
    else:
        return 0

def calc_coin(ind,result):
    num_coin=result.getNumCollectedTileCoins()
    if(num_coin>0):
        return 1
    else:
        return 0
