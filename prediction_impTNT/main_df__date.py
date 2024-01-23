
import glob
import os
import pickle
import re
import numpy as np
import pandas as pd
from function_features import *
import sys
from choose_version import choose, chooselabel
from constants import VARIABLE
import h5py
import argparse
import sklearn
import ast
from matplotlib import pyplot as plt
import time as time_package
from matplotlib.colors import LinearSegmentedColormap
import itertools
import seaborn as sns
from itertools import groupby
from operator import itemgetter
import os.path
import functools

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action='ignore', category=FutureWarning)

def static_bend(Df, limit_velocity,limit_ratio,mean_before):
    List_velocity = np.array(ast.literal_eval(Df['head_velocity']))
    Number_velocity_under = len(List_velocity[List_velocity < limit_velocity*mean_before])
    if Number_velocity_under/Df['number_time_step'] >= limit_ratio:
            return 'Static bend'

    return 'Head-Cast'


def function_angle_arc(a,b,c):
    ba = a - b
    bc = c - b
    angle  = np.arctan2(ba[1]*bc[0]-ba[0]*bc[1],ba[0]*ba[0]+bc[1]*bc[1])%np.pi

    if angle >= np.pi/2:
        angle = (angle - np.pi)
    return angle


def correction_hunch_first(pred_larva):

    Hunch = pred_larva.loc[pred_larva['prediction'] =='Hunch']
    group_time  = plateaus(Hunch[(Hunch['time']>60.0)&(Hunch['time']<90.0)]['time'], atol=0.2)

    for i in range(len(group_time)-1):
        if (group_time[i+1][0]-group_time[i][-1] < 0.75 and len(group_time[i+1]) > 10) or (group_time[i+1][0]-group_time[i][-1] < 0.4):
            pred_larva.loc[(pred_larva['time']>group_time[i][-1])&(pred_larva['time']<group_time[i+1][0]),'prediction'] = 'Hunch'
        else :
            pred_larva.loc[pred_larva['time'].isin(group_time[i+1]),'prediction'] = 'Bend'


    return pred_larva


def correction_hunch(pred_larva,tresh_hunch):
    group_time  = plateaus(pred_larva.loc[pred_larva['prediction'] =='Hunch']['time'], atol=0.2)
    if len(group_time)>0:

        for group in group_time:
            if len(group)>2:
                idx = pred_larva.loc[pred_larva['time'].isin(group)].index
                l_max = max(pred_larva.loc[idx,'larva_length'])
                l_min = min(pred_larva.loc[idx,'larva_length'])
                delta = (l_max-l_min)
                if delta<tresh_hunch:
                    pred_larva.loc[idx,'prediction'] = 'small_motion'

    return pred_larva

def new_correction(DataFrame):
    group_time  = plateaus(DataFrame['time'], atol=0.2)
    merged = []
    for G_time in group_time:
        G = []
        G_flat = list(DataFrame.loc[DataFrame['time'].isin(G_time)]['prediction'])
       # print(G_flat)
        for key, group in groupby(G_flat, key=itemgetter(1)):
            G.append(list(group))
        if len(G)>1:
            G_after = []
            while G_flat != G_after:
              #  print(G_after,G_flat)
                G_flat = [item for sublist in G for item in sublist]
                G = []
                for key, group in groupby(G_flat, key=itemgetter(1)):
                    G.append(list(group))
                for i in range(0,len(G)):

                    if len(G[i])<4:
                       # print(i,len(G)-1)
                        if i!=0 and i<len(G)-1:
                            if G[i-1][0] == G[i+1][0]:
                                G[i][:] = [G[i+1][0]]*len(G[i])

                for i in range(0,len(G)):
                    if len(G[i])<4:
                        if i not in [0,1,len(G)-1]:
                           # print(i)
                            if G[i-2][0] == G[i+1][0]:
                                G[i][:] = [G[i+1][0]]*len(G[i])
                                if len(G[i-1])<4:
                                    G[i-1][:] = [G[i+1][0]]*len(G[i-1])
                        if i not in [0,len(G)-2,len(G)-1]:
                            if G[i-1][0] == G[i+2][0]:
                                G[i][:] = [G[i+2][0]]*len(G[i])
                                if len(G[i+1])<4:
                                    G[i+1][:] = [G[i+2][0]]*len(G[i+1])
                G_after = [item for sublist in G for item in sublist]
           # print(G_after)
            if len(list(DataFrame.loc[DataFrame['time'].isin(G_time)]['prediction'])) != len(G_after):
                print('error first')
            G_after = []
           # print('g before truc',G)
            while G_flat != G_after:
             #   print('Gflat,Gafter',G_flat,G_after)
                G_flat = [item for sublist in G for item in sublist]
                G = []
                for key, group in groupby(G_flat, key=itemgetter(1)):
                    G.append(list(group))
                #print('G',G)
                for i in range(0,len(G)):
                    #print(i,G,len(G),G[i])
                    if len(G[i])<4 and len(G[i])!=len(G_flat):
                        if i ==0 :
                           # print(i,G[i][:])
                            G[i][:] = [G[i+1][0]]*len(G[i])
                            #print('inside',i,G[i][:])
                        if i ==len(G)-1 :
                           # print('i',i)
                            G[i][:] = [G[i-1][0]]*len(G[i])
                            #print('last',i,G[i][:],)
                G_after = [item for sublist in G for item in sublist]

            G_flat = [item for sublist in G for item in sublist]
            G = []
            for key, group in groupby(G_flat, key=itemgetter(1)):
                G.append(list(group))

            for i in range(0,len(G)):
                if len(G[i])<4 and len(G[i])!=len(G_flat):
                    G[i][:] = [G[i+1][0]]*len(G[i])
        merged.extend(list(itertools.chain.from_iterable(G)))
    return merged


def correction(DataFrame):
    group_time  = plateaus(DataFrame['time'], atol=0.2)

    merged = []
    for G_time in group_time:

        G = []
        for key, group in groupby((DataFrame.loc[DataFrame['time'].isin(G_time)]['prediction']), key=itemgetter(1)):
            G.append(list(group))

        if len(G)>1:


            for i in range(1,len(G)-1):
                if len(G[i])<4:
                    if G[i-1][0] == G[i+1][0]:
                        G[i][:] = [G[i+1][0]]*len(G[i])

            for i in range(2,len(G)-2):
                if len(G[i])<4:
                    if G[i-1][0] == G[i+1][0]:
                        G[i][:] = [G[i+1][0]]*len(G[i])

                    elif G[i-1][0] == G[i+2][0]:
                        G[i][:] = [G[i+2][0]]*len(G[i])
                    elif G[i][0] == G[i+2][0] and len(G[i+1])<4:
                        G[i+1][:] = [G[i+2][0]]*len(G[i+1])

                    elif G[i-2][0] == G[i+1][0]:
                        G[i][:] = [G[i+1][0]]*len(G[i])
                    elif G[i][0] == G[i-2][0] and len(G[i-1])<4:
                        G[i-1][:] = [G[i-2][0]]*len(G[i-1])
                    else :
                        G[i][:] = [G[i-1][0]]*len(G[i])

            for j in [i for i in range(2) if len(G)>i+1]:
                if len(G[j])<4 :
                  #  print(G[j+1])
                    G[j][:] = [G[(j+1)][0]]*len(G[j])

            for j in [-i for i in [2,1] if len(G)>i]:
                if len(G[j])<4:
                    G[j][:] = [G[(j-1)][0]]*len(G[j])

        merged.extend(list(itertools.chain.from_iterable(G)))

    return merged

def plateaus(A,atol):
    runs = np.split(A, np.where(np.abs(np.diff(A)) >= atol)[0] + 1)
    return [list(x) for x in runs]


def main(arg_str):
    print(sklearn.__version__)
    version = 20180409
    arg_parser = argparse.ArgumentParser(
        description='Larvae behavior analysis')

    arg_parser.add_argument(
        '-v', '--version', action='version', version='%(prog)s ' + str(version))
    arg_parser.add_argument('--id', required=True, action='store', type=int,
                            help='A simulation identifier to be added to the output file')
    arg_parser.add_argument('-n', '--name', action='store',
                            type=str)
    arg_parser.add_argument('-a', '--ac',
                            action='store', type=str)

    input_args = arg_parser.parse_args(arg_str.split())

    ac = input_args.ac
    id_number = int(input_args.name)

    date_pre = pd.read_csv('Date_francesca.csv',delimiter = ';', names =  ['line', 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    date_pre_ = date_pre.loc[id_number]

    Line = date_pre['line'][id_number].replace(' ','')

    print('line',id_number)

    path_out = 'output/not_all_dates/'
    os.makedirs(path_out+'/'+Line ,exist_ok=True)

    tresh_hunch = 0.4
    static_tresh_one = 1 - 0.3
    static_tresh_two = 0.75

    Date_name = '12_12_CLF_BBHR_impTNT_12_12_class_weightNone_tag_lenback_thresh_'+str(round(1-static_tresh_one,2)).split('.')[-1]+'_'+str(static_tresh_two).split('.')[-1]+'_threshold_'+str(tresh_hunch)+'_smallactions'
    t_start = 0.0
    t_end = 120.0

    Color=['#17202a', '#C70039','#8bc34a','#2e86c1', '#26c6da', '#f1c40f','#f088e6']

    labels_ = ['run_large', 'bend_large', 'stop_large',
    'hunch_large', 'back_crawl_large', 'roll_large','small_motion']

    feats = ['S_smooth_5', 'S_deriv_smooth_5', 'eig_smooth_5', 'eig_deriv_smooth_5', 'motion_velocity_norm_smooth_5',
         'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5', 'long_diff_', 'projection_head', 'projection_tail','theta_head','theta_tail']
    feats_before = ['S_smooth_5','eig_smooth_5','eig_deriv_smooth_5','head_velocity_norm_smooth_5','motion_velocity_norm_smooth_5']
    feats_ = ['S_smooth_5', 'eig_smooth_5','motion_velocity_norm_smooth_5','head_velocity_norm_smooth_5']
    Index_df = ['time','Larva','Date','Line','behav','S_smooth_5','S_deriv_smooth_5','eig_smooth_5',
     'eig_deriv_smooth_5','motion_velocity_norm_smooth_5','head_velocity_norm_smooth_5',
     'tail_velocity_norm_smooth_5','long_diff_',
     'projection_head','projection_tail','theta_head','theta_tail',
     'S_smooth_5_before','eig_smooth_5_before','eig_deriv_smooth_5_before','head_velocity_norm_smooth_5_before','motion_velocity_norm_smooth_5_before',
          'S_smooth_5_before_2','eig_smooth_5_before_2','eig_deriv_smooth_5_before_2','head_velocity_norm_smooth_5_before_2','motion_velocity_norm_smooth_5_before_2']

    Index_RF = ['S_smooth_5',
       'S_deriv_smooth_5', 'eig_smooth_5', 'eig_deriv_smooth_5',
       'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5',
       'tail_velocity_norm_smooth_5', 'long_diff_', 'projection_head',
       'projection_tail', 'theta_head', 'theta_tail', 'S_smooth_5_before',
       'eig_smooth_5_before', 'eig_deriv_smooth_5_before',
       'head_velocity_norm_smooth_5_before',
       'motion_velocity_norm_smooth_5_before', 'S_smooth_5_before_2',
       'eig_smooth_5_before_2', 'eig_deriv_smooth_5_before_2',
       'head_velocity_norm_smooth_5_before_2',
       'motion_velocity_norm_smooth_5_before_2']

    labels_before = ['run_large', 'bend_large', 'stop_large',
    'hunch_large', 'back_crawl_large', 'roll_large','small_motion','empty']

    Index_mean = ['S_smooth_5 max','S_smooth_5 min','S_deriv_smooth_5 max',
     'S_deriv_smooth_5 min','eig_smooth_5 max','eig_smooth_5 min','eig_deriv_smooth_5 max',
     'eig_deriv_smooth_5 min',
     'motion_velocity_norm_smooth_5 max',
     'motion_velocity_norm_smooth_5 min',
     'head_velocity_norm_smooth_5 max',
     'head_velocity_norm_smooth_5 mean',
     'head_velocity_norm_smooth_5 min',
     'tail_velocity_norm_smooth_5 max',
     'tail_velocity_norm_smooth_5 min','long_diff_ max',
     'long_diff_ min',
     'projection_head max',
     'projection_head min',
     'projection_tail max',
     'projection_tail min', 'theta_head max',
     'theta_head min',
     'theta_tail max',
     'theta_tail min',
    'S_smooth_5_before_mean',
     'eig_smooth_5_before_mean',
     'head_velocity_norm_smooth_5_before_mean',
     'motion_velocity_norm_smooth_5_before_mean']

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    #if not os.path.exists('output/'+str(ac)+'/'+Line+'/Predictions'+str(ac)+'_'  + Line.replace('/','_').replace('@','_') + '22_03.pkl'):

    # with open('Classifier_21_03_2.pkl', 'rb') as handle:
    #     CLF_BBH,CLF_BH,CLF_HCSO = pickle.load(handle)
    with open('CLF_BBHR_impTNT_12_12_class_weightNone_tag_lenback.pkl', 'rb') as handle:
        CLF_BBHR = pickle.load(handle)
    #with open('Classifier_Static_bend.pkl', 'rb') as handle:
    # with open('CLF_Static_bend_16_06.pkl', 'rb') as handle:
    #     CLF_Static_bend = pickle.load(handle)
    var = choose(ac)

    id = input_args.id
    path_=var.path
    print(Line)
    #if not os.path.exists(path_out+Line+'/Ethogramme_'+ Line.split('/')[0]+'_'+ Line.split('/')[-1] + Date_name+'_'+str(id_number)+'.pdf'):
    fichier_w = open('../Features_hunch_per_time/output/t2/bend_large/Data_t2_bend_large_'+Line.replace('/','')+'.pkl', 'rb')
    (Data_bend)=pickle.load(fichier_w)
    print(Data_bend.shape)
    fichier_w = open('../Features_hunch_per_time/output/t2/back_crawl_large/Data_t2_back_crawl_large_'+Line.replace('/','')+'.pkl', 'rb')
    (back_crawl_large)=pickle.load(fichier_w)
    print(back_crawl_large.shape)

    fichier_w = open('../Features_hunch_per_time/output/t2/hunch_large/Data_t2_hunch_large_'+Line.replace('/','')+'.pkl', 'rb')
    (Data_hunch)=pickle.load(fichier_w)

    fichier_w = open('../Features_hunch_per_time/output/new/t2/crawl_weak/Data_t2_crawl_weak_'+Line.replace('/','')+'.pkl', 'rb')
    (Data_run_weak)=pickle.load(fichier_w)

    #print(Data_run_weak.shape)
    fichier_w = open('../Features_hunch_per_time/output/new/t2/crawl_strong/Data_t2_crawl_strong_'+Line.replace('/','')+'.pkl', 'rb')
    (Data_run)=pickle.load(fichier_w)

    # print(Data_run.shape)
    # print(Data_run_weak.shape)
    # print(Data_hunch.shape)
    fichier_w = open('../Features_hunch_per_time/output/t2/head_retraction_weak/Data_t2_head_retraction_weak_'+Line.replace('/','')+'.pkl', 'rb')
    (Data_hunch_weak)=pickle.load(fichier_w)
    
    Data_hunch_weak.loc['behav'] = 'hunch_weak'
    Data_hunch.loc['behav'] = 'hunch_large'
    back_crawl_large.loc['behav'] = 'back_crawl_large'
    Data_bend.loc['behav'] = 'bend_large'
    Data_run_weak.loc['behav'] = 'run_weak'
    Data_run.loc['behav'] = 'run'
    #Data_predic  = pd.concat([Data_hunch,Data_hunch_weak,Data_bend,back_crawl_large],axis=1)
    Data_predic  = pd.concat([Data_hunch,Data_hunch_weak,Data_bend,back_crawl_large,Data_run,Data_run_weak],axis=1)

    Data_predic = Data_predic.T
    Data_predic = Data_predic.reset_index(drop=True)
    date_pre_ = date_pre_.iloc[1:]
    Date_ = [x for x in list(date_pre_) if pd.notnull(x)]
    Data_predic = Data_predic.loc[Data_predic['Date'].isin(Date_)]

    #print(Date_)
    #print(Data_predic.shape)
    for date in Date_:
        print('date',date)
        name_file = path_ + Line + '/' + str(date) +'/trx.mat'
        try :
            f = h5py.File(name_file, 'r')
            trx = f.get('trx')

        except Exception as error :
            f = h5py.File(path_ + Line + '/' + str(date) +'/trx3.mat', 'r+')
            trx = f.get('trx')

        Liste_larve = [f[trx['numero_larva_num'][0][i]][0][0] for i in range(len(trx['numero_larva_num'][()][0]))]
        data_by_date = Data_predic.loc[Data_predic['Date'] == date]

        for larva in list(set(data_by_date['Larva'])):
            Data_larva = data_by_date.loc[data_by_date['Larva'] == larva]

            nb_larva = Liste_larve.index(larva)
            Liste_time = [j for j in f[trx['t'][0][nb_larva]][0]]
            Liste_behav = f[trx['global_state_large_state'][0][nb_larva]][0]-1

            idx_time = [Liste_time.index(i) for i in list(Data_larva['time'])]
            lenght = f[trx['larva_length_smooth_5'][0][nb_larva]][0][idx_time]
            Data_predic.loc[list(Data_larva.index),'larva_length'] = lenght

            idx_ball = [j for j in range(len(f[trx['ball_proba'][0][nb_larva]][0])) if f[trx['ball_proba'][0][nb_larva]][0][j]>0.8]
            if len(idx_ball)>0:
                idx_ball.extend([i for i in [idx_ball[0]-1,idx_ball[0]-2,idx_ball[-1]+1,idx_ball[-1]+2] if i < len(Liste_behav) and i>=0])
            Liste_time_ball = [Liste_time[i] for i in idx_ball]
            Data_predic.loc[Data_larva.loc[Data_larva['time'].isin(Liste_time_ball)].index,'behav'] = 'ball'
        f.close()
    print(Data_predic.shape)

    DF_without_ball = copy.deepcopy(Data_predic)
    DF_ball = DF_without_ball.loc[(DF_without_ball['behav']=='ball')]
    DF_without_ball = DF_without_ball.loc[~(DF_without_ball['behav']=='ball')]
    DF_ball['prediction'] = DF_ball['behav']

   # DF_ball.to_csv(path_out+'/'+Line+'/Ball_'+Line.replace('/','_').replace('@','_')+Date_name+'_'+str(id_number)+'.csv')


    X_pred=DF_without_ball.loc[:,Index_RF]
    DF_prediction=pd.DataFrame({'index':list(DF_without_ball.index),'time': [float(x) for x in list(DF_without_ball['time'])],'Date':list(DF_without_ball['Date']),'Line':list(DF_without_ball['Line']),'Larva':list(DF_without_ball['Larva']),'prediction':np.arange(len(X_pred)),'probabilities':np.arange(len(X_pred))}).set_index('index')

    for clf in CLF_BBHR:
        y_pred=clf.predict(X_pred)
        DF_prediction['test'+str(str(CLF_BBHR.index(clf)))]=pd.Series(list(y_pred),index=list(X_pred.index))
    for i in range(len(DF_prediction.index[:])):
        DF_prediction.iloc[i,4]=DF_prediction.iloc[i,6:].value_counts().index[0]
        DF_prediction.iloc[i,5]=DF_prediction.iloc[i,6:].value_counts()[0]/sum(DF_prediction.iloc[i,6:].value_counts())

    Prediction_after_correction = copy.deepcopy(DF_prediction)
    Prediction_after_correction = pd.concat([Prediction_after_correction,DF_ball.loc[:,list(Prediction_after_correction.columns)[:5]]])
    Prediction_after_correction = pd.concat([Prediction_after_correction.loc[:,['time','Date','Line','Larva','prediction']] ,Data_predic.loc[:,['larva_length']]],axis=1)




    # if len(Data_predic.loc[~Data_predic['larva_length'].isna()]) != len(Data_predic):
    #     print(id_number, 'errors')

    for date in set(Prediction_after_correction['Date']):
        pred_date = Prediction_after_correction[Prediction_after_correction['Date'] == date]
        for larva in set(pred_date['Larva']):
            pred_larva_ = pred_date[pred_date['Larva'] == larva]
            if len(set(pred_larva_['prediction']))>1:
                pred_larva = pred_larva_.sort_values(by = 'time')
                pred_larva = correction_hunch_first(pred_larva)
                Prediction_after_correction.loc[pred_larva.index,'prediction'] = pred_larva['prediction']

    for date in set(Prediction_after_correction['Date']):
        pred_date = Prediction_after_correction[Prediction_after_correction['Date'] == date]
        for larva in set(pred_date['Larva']):
            pred_larva_ = pred_date[pred_date['Larva'] == larva]
            if len(set(pred_larva_['prediction']))>1:
                pred_larva = pred_larva_.sort_values(by = 'time')
                pred_larva = correction_hunch(pred_larva,tresh_hunch)
                Prediction_after_correction.loc[pred_larva.index,'prediction'] = pred_larva['prediction']

    for date in set(Prediction_after_correction['Date']):
        pred_date = Prediction_after_correction[Prediction_after_correction['Date'] == date]
        for larva in set(pred_date['Larva']):
            pred_larva_ = pred_date[pred_date['Larva'] == larva]
            if len(set(pred_larva_['prediction']))>1:
                pred_larva = pred_larva_.sort_values(by = 'time')
                #pred_larva['prediction'] = correction(pred_larva)
                pred_larva['prediction'] =  new_correction(pred_larva)
                Prediction_after_correction.loc[pred_larva.index,'prediction'] = pred_larva['prediction']

    Hunch_before = Prediction_after_correction.loc[Prediction_after_correction['prediction']=='Hunch']
    not_same = pd.concat([Hunch_before,Prediction_after_correction.loc[Hunch_before.index]]).drop_duplicates(keep=False)

    Prediction_after_correction.loc[Prediction_after_correction['prediction']=='Hunch'].to_csv(path_out+'/'+Line+'/PREDICTION_Hunch_'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')
    Prediction_after_correction.loc[Prediction_after_correction['prediction']=='Back'].to_csv(path_out+'/'+Line+'/PREDICTION_Back_'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')
    Prediction_after_correction.loc[Prediction_after_correction['prediction']=='small_motion'].to_csv(path_out+'/'+Line+'/PREDICTION_Small_'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')
    Prediction_after_correction.loc[Prediction_after_correction['prediction']=='Run'].to_csv(path_out+'/'+Line+'/PREDICTION_Runl_'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')

    #Prediction_after_correction = pd.concat([Prediction_after_correction,DF_ball.loc[:,list(Prediction_after_correction.columns)[:5]]])

    pd.concat([Prediction_after_correction.loc[:,['time','Date','Line','Larva','prediction']] ,Data_predic.loc[:,['time','larva_length']]],axis=1).to_csv(path_out+'/'+Line+'/PREDICTION_before_'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')
    #Prediction_after_correction.to_csv(path_out+'/'+Line+'/Prediction_after_correction'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')

    nb_larva_all = -1
    df_behav = pd.DataFrame({'index':np.arange(0,1200,1)}).set_index('index')

    for date in Date_:

        name_file = path_ + Line + '/' + str(date) +'/trx.mat'

        try :
            f = h5py.File(name_file, 'r')
            trx = f.get('trx')

        except Exception as error :
            f = h5py.File(path_ + Line + '/' + str(date) +'/trx3.mat', 'r+')
            trx = f.get('trx')
        for nb_larva in range(len(trx['numero_larva_num'][()][0])):

            Time = []
            Behav = []
            nb_larva_all+=1
            larva = f[trx['numero_larva_num'][0][nb_larva]][0][0]
            Liste_behav = f[trx['global_state_large_state'][0][nb_larva]][0]-1
            Liste_time = [round(j*10) for j in f[trx['t'][0][nb_larva]][0] if round(j*10)<1200]

            #II_index = [Liste_time.index(i) for i in Liste_time if i>570 and i<900]

            groupy = []
            for k, g in groupby(enumerate(Liste_time), lambda x: x[0]-x[1]):
                groupy.append(list(map(int,map(itemgetter(1), g))))

            for i in range(len(groupy)-1):
                Liste_ = groupy[i]
                Time.extend(Liste_)
                Behav.extend(Liste_behav[[Liste_time.index(i) for i in Liste_]])

                if Time[-1]!=groupy[i+1][0]:
                    liste_tps = np.arange(Time[-1],groupy[i+1][0])
                    Time.extend(liste_tps)
                    #Behav.extend([Behav[-1]]*len(liste_tps))
                    Behav.extend([7]*len(liste_tps))
            df_behav.loc['Line',nb_larva_all] = Line
            df_behav.loc['date',nb_larva_all] = date
            df_behav.loc['larva',nb_larva_all] = larva
            df_behav.loc[Time,nb_larva_all] = [labels_before[int(i)] for i in Behav]
        f.close()
    DF = df_behav.T
    df_behav_before = copy.deepcopy(df_behav)

    Prediction_bend = Prediction_after_correction.loc[Prediction_after_correction['prediction'] == 'Bend']
    Features_bend = Data_predic.loc[Prediction_bend.index]

    idx = -1

    Features_mean = pd.DataFrame({})
        # print(Features_by_date.loc[Features_by_date['time'] < 60.0]['head_velocity_norm_smooth_5'])

    for date in Date_ :
        name_file = path_ + Line + '/' + str(date) +'/trx.mat'
        try :
            f = h5py.File(name_file, 'r')
            trx = f.get('trx')

        except Exception as error :
            f = h5py.File(path_ + Line + '/' + str(date) +'/trx3.mat', 'r+')
            trx = f.get('trx')

        Features_by_date = Features_bend.loc[Features_bend['Date'] == date]
        velocity_before_group = np.mean(Features_by_date.loc[Features_by_date['time'] < 60.0]['motion_velocity_norm_smooth_5'])
        velocity_before_group_head = np.mean(Features_by_date.loc[Features_by_date['time'] < 60.0]['head_velocity_norm_smooth_5'])

        Liste_larve = [f[trx['numero_larva_num'][0][i]][0][0] for i in range(len(trx['numero_larva_num'][()][0]))]

        for larva in list(set(Features_by_date['Larva'])):

            nb_larva = Liste_larve.index(larva)
            Features_larva = Features_by_date.loc[Features_by_date['Larva'] == larva]
            Features_larva = Features_larva.sort_values(by = 'time')
            group_time  = plateaus(Features_larva['time'], atol=0.2)

            if (len(Features_larva.loc[Features_larva['time'] < 60.0]))!=0:
                velocity_before = np.mean(Features_larva.loc[Features_larva['time'] < 60.0]['motion_velocity_norm_smooth_5'])
                velocity_before_head = np.mean(Features_larva.loc[Features_larva['time'] < 60.0]['head_velocity_norm_smooth_5'])
                #print('larva',velocity_before_head)
            else :
                velocity_before = velocity_before_group
                velocity_before_head = velocity_before_group_head
            for group in group_time:
                idx += 1
                DF = Features_larva.loc[Features_larva['time'].isin(group)]
                Liste_time = [j for j in f[trx['t'][0][nb_larva]][0]]
                idx_time = [Liste_time.index(i) for i in list(DF['time'])]

                Features_mean.loc[idx,'Larva'] = larva
                Features_mean.loc[idx,'Line'] = Line
                Features_mean.loc[idx,'Date'] = date
                Features_mean.loc[idx,'time_start'] = DF['time'].iloc[0]
                Features_mean.loc[idx,'time_end'] = DF['time'].iloc[-1]
                #
                # Features_mean.loc[idx, 'motion_velocity_norm_smooth_5 max'] = max(DF['motion_velocity_norm_smooth_5']) - velocity_before
                # Features_mean.loc[idx, 'motion_velocity_norm_smooth_5 min'] = min(DF['motion_velocity_norm_smooth_5']) - velocity_before
                # Features_mean.loc[idx, 'motion_velocity_norm_smooth_5 mean'] = np.mean(DF['motion_velocity_norm_smooth_5'] - velocity_before)

                # Features_mean.loc[idx, 'head_velocity_norm_smooth_5 max'] = max(DF['head_velocity_norm_smooth_5']) - velocity_before_head
                # Features_mean.loc[idx, 'head_velocity_norm_smooth_5 min'] = min(DF['head_velocity_norm_smooth_5']) - velocity_before_head
                
                # Features_mean.loc[idx, 'head_velocity_norm_smooth_5 mean'] = np.mean(DF['head_velocity_norm_smooth_5']) - velocity_before_head

                Features_mean.loc[idx, 'velocity_before_head'] =  velocity_before_head
                Features_mean.loc[idx, 'velocity_before'] =  velocity_before

                Features_mean.loc[idx,'head_velocity'] = str(list(DF['head_velocity_norm_smooth_5']))

                #-velocity_before_head))
                # Features_mean.loc[idx,'head_velocity var'] = np.var(DF['head_velocity_norm_smooth_5'])
                
                # x_head = f[trx['x_head'][0][nb_larva]][0][idx_time]
                # y_head = f[trx['y_head'][0][nb_larva]][0][idx_time]
                # x_tail = f[trx['x_tail'][0][nb_larva]][0][idx_time]
                # y_tail = f[trx['y_tail'][0][nb_larva]][0][idx_time]
                # x_head_after = f[trx['x_head'][0][nb_larva]][0][idx_time[1:]]
                # y_head_after = f[trx['y_head'][0][nb_larva]][0][idx_time[1:]]

                # y_head = f[trx['y_head'][0][nb_larva]][0][idx_time]
                # x_center = f[trx['x_center'][0][nb_larva]][0][idx_time]
                # y_center = f[trx['y_center'][0][nb_larva]][0][idx_time]

                # # Liste_angle = np.asarray([function_angle_arc(np.array([x_head_after[i],y_head_after[i]]),np.array([x_center[i],y_center[i]]),np.array([x_head[i],y_head[i]])) for i in range(len(x_head)-1)])
                # # Delta_time = [Liste_time[1:][i]-Liste_time[:-1][i] for i in range(len(Liste_time)-1)]
                # # Velocity_angle = Liste_angle/Delta_time

                # Liste_angle = np.asarray([function_angle_arc(np.array([x_tail[i],y_tail[i]]),np.array([x_center[i],y_center[i]]),np.array([x_head[i],y_head[i]])) for i in range(len(x_head))])

                # #Features_mean.loc[idx, 'Angle_velocity_mean'] = max([abs(i) for i in Velocity_angle])

                # Features_mean.loc[idx, 'Angle max'] = max([abs(i) for i in Liste_angle])
                # Features_mean.loc[idx, 'Angle mean'] = np.mean([abs(i) for i in Liste_angle])

                # Features_mean.loc[idx, 'Angle min'] = min([abs(i) for i in Liste_angle])
                # Features_mean.loc[idx, 'Angles'] = str(Liste_angle)
                # Features_mean.loc[idx,'Angles var'] = np.var(Liste_angle)
                
                # Features_mean.loc[idx, 'number_switch'] = len([list(ele) for idx, ele in itertools.groupby(Liste_angle, lambda a: a > 0)])
                Features_mean.loc[idx, 'number_time_step'] = len(list(DF['head_velocity_norm_smooth_5']))
                
                # for feat in [i for i in feats if 'motion_velocity_norm_smooth_5' not in i and 'head_velocity_norm_smooth_5' not in i]:
                #     Features_mean.loc[idx, feat+' max']=max(DF[feat])
                #     Features_mean.loc[idx, feat+' min']=min(DF[feat])

                # for feat in feats_:
                #     Features_mean.loc[idx, feat+'_before_mean']=np.mean(DF[feat+'_before']+DF[feat+'_before_2'])

        f.close()
    #Features_mean.to_csv(path_out + Line + '/Features_mean_' + Line.replace('/','_').replace('@','_') +Date_name+str(id_number)+ '.csv')

    Features_mean = Features_mean.fillna(0)
    for idx in Features_mean.index:
        Features_mean.loc[idx,'prediction'] = static_bend(Features_mean.loc[idx], static_tresh_one,static_tresh_two,Features_mean.loc[idx,'velocity_before_head'])

    DF_prediction_bend = Features_mean.loc[:,['Larva', 'Line', 'Date','time_start', 'time_end','prediction']]
    DF_prediction_bend.to_csv(path_out+Line+'/PREDICTION_Bends_'+Line.replace('/','_').replace('@','_')+Date_name+str(id_number)+'.csv')

    Prediction_hunch = Prediction_after_correction.loc[Prediction_after_correction['prediction'] == 'Hunch']

    Features_mean_hunch = pd.DataFrame({})
    for date in Date_ :
        Features_by_date = Prediction_hunch.loc[Prediction_hunch['Date'] == date]
        for larva in list(set(Features_by_date['Larva'])):
            Features_larva = Features_by_date.loc[Features_by_date['Larva'] == larva]
            Features_larva = Features_larva.sort_values(by = 'time')
            group_time  = plateaus(Features_larva['time'], atol=0.2)

            for group in group_time:
                idx += 1
                DF = Features_larva.loc[Features_larva['time'].isin(group)]

                Features_mean_hunch.loc[idx,'Larva'] = larva
                Features_mean_hunch.loc[idx,'Line'] = Line
                Features_mean_hunch.loc[idx,'Date'] = date
                Features_mean_hunch.loc[idx,'time_start'] = DF['time'].iloc[0]
                Features_mean_hunch.loc[idx,'time_end'] = DF['time'].iloc[-1]
                l_max = max(DF['larva_length'])
                l_min = min(DF['larva_length'])
                delta = (l_max-l_min)
                Features_mean_hunch.loc[idx,'delta'] = delta

    Features_mean_hunch.to_csv(path_out + Line + '/Features_mean_hunch' + Line.replace('/','_').replace('@','_') +Date_name+str(id_number)+ '.csv')


    prediction = Prediction_after_correction.loc[(Prediction_after_correction['prediction']!='Bend')]

    for date in Date_:
        pre = prediction[prediction['Date'] == date]

        for data in pre.index:

            time = round(pre.loc[data,'time']*10)
            if time <1200:
                if pre.loc[data]['prediction'] != 'nan':
                    df_behav.loc[time,(df_behav.loc['Line'] == Line)&(df_behav.loc['date']==date)&(df_behav.loc['larva']==pre.loc[data,'Larva'])] = pre.loc[data]['prediction']
                else :
                    df_behav.loc[time,(df_behav.loc['Line'] == Line)&(df_behav.loc['date']==date)&(df_behav.loc['larva']==pre.loc[data,'Larva'])] = df_behav.loc[time-1,(df_behav.loc['Line'] == Line)&(df_behav.loc['date']==date)&(df_behav.loc['larva']==pre.loc[data,'Larva'])]

        pre_bend = DF_prediction_bend
        pre_bend = pre_bend[pre_bend['Date'] == date]

        for data in pre_bend.index:
            larva = pre_bend.loc[data,'Larva']
            for time in range(round(pre_bend.loc[data,'time_start']*10),round(pre_bend.loc[data,'time_end']*10+2)):
                if time <1200:
                    df_behav.loc[time,(df_behav.loc['Line'] == Line)&(df_behav.loc['date']==date)&(df_behav.loc['larva']==larva)] = pre_bend.loc[data]['prediction']


    df_behav = df_behav.replace('Run','run_large')
    # nb_nan = 0
    # for col in df_behav.columns:
    #     if 'empty' in list(df_behav[col]):
    #         index_ = list(df_behav[col]).index('empty')
    #         print('empty',index_,list(df_behav[col]).index('empty'))
    #         if type(index_) != int:
    #             for idx_ in index_:
    #                 nb_nan+=1
    #                 df_behav[col][idx_] = df_behav[col][idx_-1]
    #                 if df_behav[col][idx_-1] == 'empty':
    #                     df_behav[col][idx_ -1] = df_behav[col][idx_-2]
    #                     df_behav[col][idx_] = df_behav[col][idx_-1]
    #
    #         else :
    #             nb_nan+=1
    #             df_behav[col][index_] = df_behav[col][index_-1]
    #         print('empty',index_,df_behav[col][index_])
    #
    # for col in df_behav.columns:
    #     if 'bend_large' in list(df_behav[col]):
    #         index_ = list(df_behav[col]).index('bend_large')
    #         print(index_)
    #         df_behav.loc[index_,col] = df_behav.loc[index_-1,col]
    #         if df_behav.loc[index_-1,col] == 'empty':
    #             df_behav.loc[index_-1,col] = df_behav.loc[index_-2,col]
    #             df_behav.loc[index_,col] = df_behav.loc[index_-1,col]
    #         print(df_behav[col][index_],type(df_behav[col][index_]))
    # for col in df_behav.columns:
    #     if 'hunch_large' in list(df_behav[col]):
    #         index_ = list(df_behav[col]).index('hunch_large')
    #         print('hunch_large',index_,df_behav[col][index_])
    #         df_behav.loc[index_,col] = df_behav[col][index_-1]
    #         print('hunch_large',df_behav[col][index_])
    # for col in df_behav.columns:
    #     if 'back_crawl_large' in list(df_behav[col]):
    #         index_ = list(df_behav[col]).index('back_crawl_large')
    #         print('back_crawl_large',index_,df_behav.loc[index_,col])
    #         df_behav.loc[index_,col] = df_behav.loc[index_-1,col]
    #         print('back_crawl_large',df_behav[col][index_])

    # nb_nan = 0
    # for col in df_behav.columns:
    #     if 'empty' in list(df_behav[col]):
    #         index_ = list(df_behav[col]).index('empty')
    #         print('empty',index_,list(df_behav[col]).index('empty'))
    #         if type(index_) != int:
    #             for idx_ in index_:
    #                 nb_nan+=1
    #                 df_behav.loc[idx_,col] = df_behav.loc[idx_-1,col]
    #         else :
    #             nb_nan+=1
    #             df_behav.loc[index_,col] = df_behav.loc[index_-1,col]
    #         print('empty',index_,df_behav[col][index_])
    #print(df_behav.iloc[:-3,:].apply(pd.value_counts).sum(axis=1))


    nb_nan = 0
    for col in df_behav.columns:
        for indx in df_behav.index:
            if df_behav.loc[indx,col] == 'empty':
                nb_nan+=1
                df_behav.loc[indx,col] = df_behav.loc[indx-1,col]
    # print(nb_nan)
    # print(df_behav.iloc[:-3,:].apply(pd.value_counts).sum(axis=1))

    DF = df_behav.T

    DF = pd.concat([DF.iloc[:,-3:],DF.iloc[:,:-3]],axis=1)
    Col = ['Line','date','larva']+[i/10 for i in DF.columns[3:]]
    DF.columns = Col
    DF = DF.iloc[:,:1203]
    DF.to_csv(path_out + Line +'/Larva_behaviour_time_'+Line.split('/')[0]+'_'+ Line.split('/')[-1]+Date_name+str(id_number)+'.csv')

   # print(DF)
    DF_ = DF.iloc[:,4:]

    labels_p = ['small_motion', 'stop_large','run_large', 'Head-Cast','Static bend', 'Hunch',   'Back',     'roll_large',   'ball']
    Color_p = ['#B1B0B9',        '#8bc34a',    '#17202a',  '#C70039',   '#ee801b',   '#0D43C5', '#26c6da',    '#f1c40f',    '#975f1e']

    df_proba = pd.DataFrame(np.zeros((len(labels_p),len(DF_.columns))), index=labels_p, columns=DF_.columns)

    for i in DF_.columns[:]:
        df_proba.loc[:,i] = DF_.loc[:,i].value_counts()/DF_.loc[:,i].count()
    df_proba.to_csv(path_out+Line+'/Proba_data_'+Line.split('/')[0]+'_'+ Line.split('/')[-1]+ Date_name+'_'+str(id_number)+'.csv')

    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(111)
    for labels__ in labels_p:
        df = df_proba.T[labels__]
        df.plot(c=Color_p[int(labels_p.index(labels__))], linewidth=2.5, label= labels__)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r't')
    ax.set_ylabel(r'p')
    ax.set_xlim([0,120])

    ax.legend()
    plt.title(str(Date_))
    #ac+'/'+ Line +
    fig.savefig(path_out+Line+'/Probability_'+Line.split('/')[0]+'_'+ Line.split('/')[-1] + Date_name+'_'+str(id_number)+'.pdf')

    # list_numbers = [1,2,3,4,5,6,7,8]
    #
    # # DF = DF.replace('hunch_large','Hunch')
    # # DF = DF.replace('bend_large','Head-Cast')
    # # DF = DF.replace('back_crawl_large','Back')
    # DF_ = DF.replace(labels_p,list_numbers[:]).iloc[:,4:]
    #
    # DF_ = DF_.fillna(0)
    # DF_ = DF_[~(DF_.iloc[:,598:610]==0).all(axis=1)]
    # DF_ = DF_.sort_values(by = list(DF_.iloc[:,600:].columns))
    #
    # DF_ = DF_.iloc[:,570:650]
    # fig = plt.figure(figsize=((5,5)))
    # ax = fig.add_subplot(111)


    # start_time = float(60)
    # f = open(path_out + Line + '/'+'Constant_numbers'+Date_name+str(id_number)+'.txt', 'w')
    # time_windows = [1,2,5,10]

    # Cumulative = pd.DataFrame(np.zeros((len(time_windows),len(labels_p))),index = time_windows, columns = labels_p)

    # #df = DF.loc[:,str(start_time):str(start_time+5-0.1)]

    # df = df.dropna(how='all')

    # for time_window in time_windows:
    #     df = DF.iloc[:,600:round((start_time+time_window-0.1)*10)]
    #     #df = DF.loc[:,str(start_time):str(start_time+time_window-0.1)]
    #     df = df.dropna(how='all')
    # #    print(time_window,df.shape)

    #     for labels in labels_p :

    #         Cumulative.loc[time_window,labels] = (df == labels).any(axis=1).sum()/len(df)

    #     Transition = pd.DataFrame(np.zeros((len(labels_p),len(labels_p))),index = labels_p, columns= labels_p)
    #     for index in df.index:
    #         liste_group = [i[0] for i in groupby(list(df.loc[index]))]
    #         cleanedList = [x for x in liste_group if type(x) == str]
    #         for id_behav in range(len(cleanedList[:-1])):
    #             Transition.loc[cleanedList[id_behav],cleanedList[id_behav+1]] += 1
    #     nb_transition = Transition.sum().sum()
    #     nb_larva = len(df)

    #     for label in labels_p:
    #         Transition.loc[label,:] = Transition.loc[label,:]/sum(Transition.loc[label,:])
    #     Transition = Transition.fillna(0)
    #     Transition.to_csv(path_out + Line + '/Transition_'+str(time_window) +'_'+ Line.replace('/','_') +Date_name+ '_id_'+str(id_number)+'.csv')
    #     #print(Transition)
    # #print(path_out + '/' + Line + '/Cumulative_' + Line.replace('/','_') + '.csv')
    #     f.write('time window:' + str(time_window))
    #     f.write( '\n'+'number of transitions: '+str(nb_transition) + '\n' + 'nunmber of larva: '+str(nb_larva) + '\n')

    # Cumulative.to_csv(path_out  + Line + '/Cumulative_' + Line.replace('/','_') +Date_name+ '_id_'+str(id_number)+ '.csv')


    # f.close()

    labels_p =          ['run_large','Head-Cast','Static bend','stop_large','Hunch',    'Back',    'roll_large','small_motion','ball']

    Color_p = ['#ffffff','#17202a', '#C70039',      '#ee801b',  '#8bc34a', '#0D43C5', '#26c6da',    '#f1c40f',      '#B1B0B9','#975f1e']

    list_numbers = [1,2,3,4,5,6,7,8,9]

    # DF = DF.replace('hunch_large','Hunch')
    # DF = DF.replace('bend_large','Head-Cast')
    # DF = DF.replace('back_crawl_large','Back')
    DF_ = DF.replace(labels_p,list_numbers[:]).iloc[:,4:]

    DF_ = DF_.fillna(0)
    DF_ = DF_[~(DF_.iloc[:,598:610]==0).all(axis=1)]
    DF_ = DF_.sort_values(by = list(DF_.iloc[:,600:].columns))

    DF_ = DF_.iloc[:,570:650]
    fig = plt.figure(figsize=((5,5)))
    ax = fig.add_subplot(111)
    #print(DF_)
    #cmap = LinearSegmentedColormap.from_list('Custom', Color_p, len(Color_p))
    #ax = sns.heatmap(DF_, cmap=cmap, linewidths=0,cbar=False, vmin=-0.5, vmax=9.5)

    cmap = LinearSegmentedColormap.from_list('Custom', Color_p, len(Color_p))
    ax = sns.heatmap(DF_, cmap=cmap, linewidths=0,cbar=False, vmin=-0.5, vmax=9.5)
    #ax.set_yticks([])
    ax.set_ylabel('larva number')
    ax.set_xlabel('t')
    ax.set_xticks(np.arange(0, len(DF_.columns)+1, 10))
    ax.set_xticklabels([57,58,59,60,61,62,63,64,65], rotation=0, fontsize=5)
    ax.set_yticks(list(np.arange(0,len(DF_),20)))
    ax.set_yticklabels(list(np.arange(0,len(DF_),20)), rotation=0, fontsize=5)
    #ax.set_xticklabels([0,20,40,60,80,100,120,140], rotation=0, fontsize=5)
    ax.set_ylim((0,len(DF_)))
    plt.title(str(Date_))
    plt.savefig(path_out+Line+'/Ethogramme_'+ Line.split('/')[0]+'_'+ Line.split('/')[-1] + Date_name+'_'+str(id_number)+'.pdf')
    plt.show()
