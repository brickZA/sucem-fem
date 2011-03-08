from __future__ import division
import pickle, os
import numpy as N
from scipy import sparse

local_onef_1_mixed_mass = N.array([[2/3,1/3,1/3,1/6,0,0,0,0,0,0,0,0],[1/3,2/3,1/6,1/3,0,0,0,0,0,0,0,0],[1/3,1/6,2/3,1/3,0,0,0,0,0,0,0,0],[1/6,1/3,1/3,2/3,0,0,0,0,0,0,0,0],[0,0,0,0,1/6,1/12,1/12,1/24,0,0,0,0],[0,0,0,0,1/12,1/6,1/24,1/12,0,0,0,0],[0,0,0,0,1/12,1/24,1/6,1/12,0,0,0,0],[0,0,0,0,1/24,1/12,1/12,1/6,0,0,0,0],[0,0,0,0,0,0,0,0,2/27,1/27,1/27,1/54],[0,0,0,0,0,0,0,0,1/27,2/27,1/54,1/27],[0,0,0,0,0,0,0,0,1/27,1/54,2/27,1/27],[0,0,0,0,0,0,0,0,1/54,1/27,1/27,2/27]], N.float64)


global_onef_1_mixed_mass = N.array([[1/9,1/18,1/18,1/36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1/18,1/9,1/36,1/18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1/18,1/36,2/9,1/9,1/18,1/36,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1/36,1/18,1/9,2/9,1/36,1/18,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1/18,1/36,1/9,1/18,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1/36,1/18,1/18,1/9,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1/9,1/18,0,0,1/18,1/36,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1/18,1/9,0,0,1/36,1/18,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1/9,1/18,0,0,1/18,1/36,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1/18,1/9,0,0,1/36,1/18,0,0,0,0,0,0],[0,0,0,0,0,0,1/18,1/36,0,0,1/9,1/18,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1/36,1/18,0,0,1/18,1/9,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1/18,1/36,0,0,1/9,1/18,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1/36,1/18,0,0,1/18,1/9,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/9,1/18,0,1/18,1/36,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/18,2/9,1/18,1/36,1/9,1/36],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/18,1/9,0,1/36,1/18],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/18,1/36,0,1/9,1/18,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/36,1/9,1/36,1/18,2/9,1/18],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1/36,1/18,0,1/18,1/9]], N.float64)

local_onef_1_mixed_stiffness = N.array([[13/18,1/36,-7/18,-13/36,-1/2,-1/4,1/2,1/4,-2/9,-1/9,2/9,1/9],[1/36,13/18,-13/36,-7/18,-1/4,-1/2,1/4,1/2,2/9,1/9,-2/9,-1/9],[-7/18,-13/36,13/18,1/36,1/2,1/4,-1/2,-1/4,-1/9,-2/9,1/9,2/9],[-13/36,-7/18,1/36,13/18,1/4,1/2,-1/4,-1/2,1/9,2/9,-1/9,-2/9],[-1/2,-1/4,1/2,1/4,5/9,7/36,-17/36,-5/18,-1/18,1/18,-1/36,1/36],[-1/4,-1/2,1/4,1/2,7/36,5/9,-5/18,-17/36,1/18,-1/18,1/36,-1/36],[1/2,1/4,-1/2,-1/4,-17/36,-5/18,5/9,7/36,-1/36,1/36,-1/18,1/18],[1/4,1/2,-1/4,-1/2,-5/18,-17/36,7/36,5/9,1/36,-1/36,1/18,-1/18],[-2/9,2/9,-1/9,1/9,-1/18,1/18,-1/36,1/36,5/18,1/18,-7/36,-5/36],[-1/9,1/9,-2/9,2/9,1/18,-1/18,1/36,-1/36,1/18,5/18,-5/36,-7/36],[2/9,-2/9,1/9,-1/9,-1/36,1/36,-1/18,1/18,-7/36,-5/36,5/18,1/18],[1/9,-1/9,2/9,-2/9,1/36,-1/36,1/18,-1/18,-5/36,-7/36,1/18,5/18]], N.float64)

global_onef_1_mixed_stiffness = N.array([[2/3,-1/6,-1/6,-1/3,0,0,-1/3,-1/6,0,0,1/3,1/6,0,0,-1/3,-1/6,0,1/3,1/6,0],[-1/6,2/3,-1/3,-1/6,0,0,-1/6,-1/3,0,0,1/6,1/3,0,0,1/3,1/6,0,-1/3,-1/6,0],[-1/6,-1/3,4/3,-1/3,-1/6,-1/3,1/3,1/6,-1/3,-1/6,-1/3,-1/6,1/3,1/6,-1/6,-2/3,-1/6,1/6,2/3,1/6],[-1/3,-1/6,-1/3,4/3,-1/3,-1/6,1/6,1/3,-1/6,-1/3,-1/6,-1/3,1/6,1/3,1/6,2/3,1/6,-1/6,-2/3,-1/6],[0,0,-1/6,-1/3,2/3,-1/6,0,0,1/3,1/6,0,0,-1/3,-1/6,0,-1/6,-1/3,0,1/6,1/3],[0,0,-1/3,-1/6,-1/6,2/3,0,0,1/6,1/3,0,0,-1/6,-1/3,0,1/6,1/3,0,-1/6,-1/3],[-1/3,-1/6,1/3,1/6,0,0,2/3,-1/6,0,0,-1/6,-1/3,0,0,-1/3,1/3,0,-1/6,1/6,0],[-1/6,-1/3,1/6,1/3,0,0,-1/6,2/3,0,0,-1/3,-1/6,0,0,1/3,-1/3,0,1/6,-1/6,0],[0,0,-1/3,-1/6,1/3,1/6,0,0,2/3,-1/6,0,0,-1/6,-1/3,0,-1/3,1/3,0,-1/6,1/6],[0,0,-1/6,-1/3,1/6,1/3,0,0,-1/6,2/3,0,0,-1/3,-1/6,0,1/3,-1/3,0,1/6,-1/6],[1/3,1/6,-1/3,-1/6,0,0,-1/6,-1/3,0,0,2/3,-1/6,0,0,-1/6,1/6,0,-1/3,1/3,0],[1/6,1/3,-1/6,-1/3,0,0,-1/3,-1/6,0,0,-1/6,2/3,0,0,1/6,-1/6,0,1/3,-1/3,0],[0,0,1/3,1/6,-1/3,-1/6,0,0,-1/6,-1/3,0,0,2/3,-1/6,0,-1/6,1/6,0,-1/3,1/3],[0,0,1/6,1/3,-1/6,-1/3,0,0,-1/3,-1/6,0,0,-1/6,2/3,0,1/6,-1/6,0,1/3,-1/3],[-1/3,1/3,-1/6,1/6,0,0,-1/3,1/3,0,0,-1/6,1/6,0,0,2/3,-1/6,0,-1/6,-1/3,0],[-1/6,1/6,-2/3,2/3,-1/6,1/6,1/3,-1/3,-1/3,1/3,1/6,-1/6,-1/6,1/6,-1/6,4/3,-1/6,-1/3,-1/3,-1/3],[0,0,-1/6,1/6,-1/3,1/3,0,0,1/3,-1/3,0,0,1/6,-1/6,0,-1/6,2/3,0,-1/3,-1/6],[1/3,-1/3,1/6,-1/6,0,0,-1/6,1/6,0,0,-1/3,1/3,0,0,-1/6,-1/3,0,2/3,-1/6,0],[1/6,-1/6,2/3,-2/3,1/6,-1/6,1/6,-1/6,-1/6,1/6,1/3,-1/3,-1/3,1/3,-1/3,-1/3,-1/3,-1/6,4/3,-1/6],[0,0,1/6,-1/6,1/3,-1/3,0,0,1/6,-1/6,0,0,1/3,-1/3,0,-1/3,-1/6,0,-1/6,2/3]], N.float64)



# Note the more_brick_mats were calculated numerically using revision
# nmarais@sun.ac.za--femcode/newcode--main--0.2--patch-32
more_brick_mats = pickle.load(file(os.path.join(
    os.path.dirname(__file__), 'more_brick_mats_2.pickle')))
more_brick_mats.update(pickle.load(file(os.path.join(
    os.path.dirname(__file__), 'more_brick_mats_4.pickle'))))
more_brick_mats.update(pickle.load(file(os.path.join(
    os.path.dirname(__file__), 'more_brick_mats_cohen98_4.pickle'))))
more_brick_mats.update(pickle.load(file(os.path.join(
    os.path.dirname(__file__), 'more_brick_mats_cohen98_4_lumped.pickle'))))
more_brick_mats.update(pickle.load(file(os.path.join(
    os.path.dirname(__file__), 'more_brick_mats_cohen98_2_lumped.pickle'))))
more_brick_mats.update(pickle.load(file(os.path.join(
    os.path.dirname(__file__), 'more_brick_mats_cohen98_2.pickle'))))

for k in more_brick_mats.keys():
    more_brick_mats[k] = sparse.coo_matrix(more_brick_mats[k])