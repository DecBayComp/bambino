import numpy as np
import glob
import pickle
import random
import pandas as pd
import os
from os import path
import copy
import h5py
from collections import Counter
from itertools import groupby
from operator import itemgetter


def norm(A, B):
    return (((A[0] - B[0])**2 + (A[1] - B[1])**2)**(1 / 2))

def V_unit(A0, A1):
    #print('A0','a1',A0,A1)
    norm_ = norm([A0[0], A0[1]], [A1[0], A1[1]])
    if norm_!=0:
        u = [(A1[0] - A0[0]) / norm_, (A1[1] - A0[1]) / norm_]
    else :
        u=[0.0,0.0]
    return u


def director(A, B):
    if (A[0] - B[0])!=0:
        return (A[1] - B[1]) / (A[0] - B[0])
    else :
        return 0


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def vect_ortho(A, B):
    def y(x): return (((B[0] - A[0]) * (B[0] - x)) / (B[1] - A[1]) + B[1])
    return y


def coord_origin(A, B):
    return A[1] - director(A, B) * A[0]


def points_ortho(vec, X):
    x = (X[0] + director(vec[0], vec[1]) * (X[1] -
                                            coord_origin(vec[0], vec[1]))) / (1 + director(vec[0], vec[1])**2)
    y = director(vec[0], vec[1]) * x + coord_origin(vec[0], vec[1])
    return [x, y]


def projection_v(Spine, vecteur_unit_head, vecteur_unit_tail, H, T, velocity_head, velocity_tail, nb_points):
    Vec_tail = [velocity_tail * vecteur_unit_tail[0] +
                T[0], velocity_tail * vecteur_unit_tail[1] + T[1]]
    Vec_head = [velocity_head * vecteur_unit_head[0] +
                H[0], velocity_head * vecteur_unit_head[1] + H[1]]


    if True not in np.isnan(Vec_head):
        Norm_head = 0
        S = copy.deepcopy(list(Spine))
        a = 0
        for n in range(0, nb_points - 1):
            if points_ortho([Spine[n + 1], Spine[n]], Vec_head)[0] < max([Spine[n][0], Spine[n + 1][0]]) and points_ortho([Spine[n + 1], Spine[n]], Vec_head)[0] > min([Spine[n][0], Spine[n + 1][0]]):
                S.insert(
                    n + a, points_ortho([Spine[n + 1], Spine[n]], Vec_head))
                a += 1

        Mini = [norm(S[i], Vec_head) for i in range(len(S))].index(
            min([norm(S[i], Vec_head) for i in range(len(S))]))
        if Mini == 0:
            Norm_head -= norm(Vec_head,
                              points_ortho([Spine[1], Spine[0]], Vec_head))
        else:
            for n in range(0, Mini):
                Norm_head += norm(S[n], S[n + 1])
    else:
        Norm_head = 0.0

    if True not in np.isnan(Vec_tail):
        Norm_tail = 0
        S = copy.deepcopy(list(Spine)[::-1])
        a = 0
        for n in range(0, nb_points - 1):
            if points_ortho([S[n + 1 + a], S[n + a]], Vec_tail)[0] < max([S[n + a][0], S[n + 1 + a][0]]) and points_ortho([S[n + 1 + a], S[n + a]], Vec_tail)[0] > min([S[n + a][0], S[n + 1 + a][0]]):
                S.insert(
                    n + a, points_ortho([S[n + 1 + a], S[n + a]], Vec_tail))
                a += 1
        Mini = [norm(S[i], Vec_tail) for i in range(len(S))].index(
            min([norm(S[i], Vec_tail) for i in range(len(S))]))
        if Mini == 0:
            Norm_tail -= norm(Vec_tail, points_ortho([S[1], S[0]], Vec_tail))
        else:
            for n in range(0, Mini):
                Norm_tail += norm(S[n], S[n + 1])
    else:
        Norm_tail = 0.0
    return Norm_head, Norm_tail


def potential_div(data,f, time, Length,numero_index):
    nb_points = len(f[data['y_spine'][()][0][numero_index]][()])
    Spine = np.zeros((nb_points, 2))
    Spine[:, 0] = [f[data['x_spine'][()][0][numero_index]][()][i][time]
                   for i in range(nb_points)]
    Spine[:, 1] = [f[data['y_spine'][()][0][numero_index]][()][i][time]
                   for i in range(nb_points)]

    G = [f[data['x_center'][()][0][numero_index]][()][0][time],
         f[data['y_center'][()][0][numero_index]][()][0][time]]
    H = [f[data['x_head'][()][0][numero_index]][()][0][time],
         f[data['y_head'][()][0][numero_index]][()][0][time]]
    T = [f[data['x_tail'][()][0][numero_index]][()][0][time],
         f[data['y_tail'][()][0][numero_index]][()][0][time]]
    try:
        H1 = [f[data['x_head'][()][0][numero_index]][()][0][time + 1],
              f[data['y_head'][()][0][numero_index]][()][0][time + 1]]
        T1 = [f[data['x_tail'][()][0][numero_index]][()][0][time + 1],
              f[data['y_tail'][()][0][numero_index]][()][0][time + 1]]
    except:
        H1 = [f[data['x_head'][()][0][numero_index]][()][0][time - 1],
              f[data['y_head'][()][0][numero_index]][()][0][time - 1]]
        T1 = [f[data['x_tail'][()][0][numero_index]][()][0][time - 1],
              f[data['y_tail'][()][0][numero_index]][()][0][time - 1]]

    vecteur_unit_head = V_unit(H, H1)
    vecteur_unit_tail = V_unit(T, T1)
    velocity_head = f[data['head_velocity_norm_smooth_5'][()][0][numero_index]][0][time]
    velocity_tail = f[data['tail_velocity_norm_smooth_5'][()][0][numero_index]][0][time]

    Div_head_tail = ((norm(G, H)) / (norm(G, T)))
    proj_head, proj_tail = projection_v(
        Spine, vecteur_unit_head, vecteur_unit_tail, H, T, velocity_head, velocity_tail, nb_points)
    x = (norm(T, Spine[-2, :]) * norm(T, T1))
    cos_theta_tail = ((((T[0] - Spine[-2, 0]) * (T[0] - T1[0])) + (
        (T[1] - Spine[-2, 1]) * (T[1] - T1[1])))/(x or not x))
    x2 = (norm(H, Spine[1, :]) * norm(H, H1))
    cos_theta_head = ((((H[0] - Spine[1, 0]) * (H[0] - H1[0])) + (
        (H[1] - Spine[1, 1]) * (H[1] - H1[1])))/(x2 or not x2))

    return Div_head_tail, proj_head, proj_tail, cos_theta_head, cos_theta_tail


def obtaining_features(trx,f,df,Int,behav_number,t_start,t_end,feats,feats_before):
    NB_LARVA = 0
    for nb_larva in range(len(trx['numero_larva_num'][()][0])):
        Larva = f[trx['numero_larva_num'][0][nb_larva]][0][0]
        Line = f[trx['neuron'][0][nb_larva]][:].tobytes().decode('UTF-16')+'_'+f[trx['stimuli'][0][nb_larva]][:].tobytes().decode('UTF-16')+'_'+f[trx['protocol'][0][nb_larva]][:].tobytes().decode('UTF-16')
        Date = f[trx['id'][0][nb_larva]][:].tobytes().decode('UTF-16')
        Index_time = list(np.where((f[trx['t'][0][nb_larva]][0]>t_start) & (f[trx['t'][0][nb_larva]][0]<t_end))[0])

        if len(Index_time)>0:
            NB_LARVA+=1
            Index_behav = np.where(f[trx['global_state_large_state'][0][nb_larva]][0] == behav_number)[0]
            Index_inter = list(set(Index_time) & set(Index_behav))
            if len(Index_inter)> 0:
                Length = sum(f[trx['larva_length_smooth_5'][0][nb_larva]][0][:])/len(f[trx['larva_length_smooth_5'][0][nb_larva]][0][:])
                for k, g in groupby(enumerate(sorted(Index_inter)), lambda x : x[0] - x[1]):
                    #Int+=0
                    Index = list(dict(g).values())

                    dic = {'projection_tail':[],'long_diff_':[],'projection_head':[],'theta_head':[],'theta_tail':[]}
                    for idx in Index :
                        Div, projection_head, projection_tail, theta_head, theta_tail = potential_div(
                                trx,f, idx, Length,nb_larva)
                        dic['projection_tail'].append(projection_tail)
                        dic['long_diff_'].append(Div)
                        dic['projection_head'].append(projection_head)
                        dic['theta_head'].append(theta_head)
                        dic['theta_tail'].append(theta_tail)
                        df.loc['time',Int+int(Index.index(idx))] = f[trx['t'][0][nb_larva]][0][idx]
                        #str([f[trx['t'][0][nb_larva]][0][Index[0]],f[trx['t'][0][nb_larva]][0][Index[-1]]])
                        df.loc[['Line','Larva','Date'],Int+int(Index.index(idx))] = [Line,Larva,Date]

                    #df.iloc[:,slice(Int,Int+len(Index))] = 0
                    for feat in feats[-5:]:
                        # print(dic[feat])
                        # print(len(dic[feat]))
                        # print('len(Index)',len(Index))
                        # print('feat',feat)
                        # print(df)
                        # print(df.loc[feat,Int:Int+len(Index)])
                        # print(list(dic[feat]))
                        df.loc[feat,Int:Int+len(Index)] = list(dic[feat])

                    for feat in feats[:-5]:
                        df.loc[feat,Int:Int+len(Index)] = f[trx[feat][0][nb_larva]][0][Index]
                    for feat in feats_before:
                        df.loc[feat+'_before',Int:Int+len(Index)] = f[trx[feat][0][nb_larva]][0][Index[0]-1]
                        df.loc[feat+'_before_2',Int:Int+len(Index)] = f[trx[feat][0][nb_larva]][0][Index[0]-2]
                    #print(df.loc[:,Int:Int+len(Index)])
                    Int = Int+len(Index)
    return df,Int
