from river import rules, tree, datasets, drift, metrics, evaluate
from IPython import display

import random
import functools
from itertools import combinations, product


import numpy as np
import time

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import fatf.utils.data.datasets as fatf_datasets

import fatf.fairness.data.measures as fatf_dfm

import fatf.utils.data.tools as fatf_data_tools

from tqdm import tqdm
from matplotlib import pyplot as plt

from classes import *

def get_index (attribute, attr_list):
    for j in range(len(attr_list)):
        if attr_list[j] == attribute:
            return j
        
def percentage(value, all_records):
    return round((value * all_records) / 100)

def ideal_record_test(rec, rule_att, rule_value):
    if rec[rule_att] > rule_value:
        return True
    else:
        return None
    
def get_value_swap_records(x, processed, protected, attr_list):
    
    protected_inx = []
    for att in protected:
        protected_inx.append(get_index(att, attr_list))
                
    current = list(x.values())
    vs_records = []
    check = True
    vs_decision = None
    
    for record in list(processed.keys()):
        check = True
        for i in range(len(record)):
            if i not in protected_inx:
                if record[i] != current[i]:
                    check = False
            else:
                if record[i] == current[i]:
                    check = False
        if check:
            vs_records.append(record)
            vs_decision = processed[record]['decision']
    
    return vs_records, vs_decision

def get_fairness(model, protected, processed, protected_values):
    PP, PN, DP, DN = [], [], [], []
    PP_c, PN_c, DP_c, DN_c = 0, 0, 0, 0

    for rec in list(processed.keys()):
        og_rec = processed[rec]['dict_form']
        proba = model.predict_proba_one(og_rec)[True]
        if processed[rec]['decision'] == True:
            if processed[rec]['dict_form'][protected[0]] == protected_values[0]:
                PP_c = PP_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    PP.append(((proba, rec)))
            else:
                DP_c = DP_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    DP.append(((proba, rec)))
        else:
            if processed[rec]['dict_form'][protected[0]] == protected_values[0]:
                PN_c = PN_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    PN.append(((proba, rec)))
            else:
                DN_c = DN_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    DN.append(((proba, rec)))
                  
    try:
        fairness = (PP_c) / ((PP_c)+(PN_c)) - (DP_c) / ((DP_c)+(DN_c))
    except:
        fairness = 0
    
    if fairness != 0:
        fair_number = round(((DP_c)+(DN_c)) * ((PP_c)+(DP_c)) / ((PP_c)+(PN_c)+(DP_c)+(DN_c)))
        
    if fairness < 0:
        DN = PN
        PP = DP

    DN = [e for e in DN if e[0] > 0.5]
    PP = [e for e in PP if e[0] < 0.5]
    
    DN = sorted(DN, reverse=True)
    PP = sorted(PP)
    
    return DN, PP, fairness

def evaluation_human (processed, protected, Y, attr_list):
    DN, DP, PN, PP = 0, 0, 0, 0
    Y_final = []

    for r in processed.keys():
        
        record = processed[r]['dict_form']
        sa = record[protected[0]]
        decision = processed[r]['decision']
        
        Y_final.append(decision) #for accuracy

        if decision == 0:
            if sa == 0:
                PN = PN + 1
            else:
                DN = DN + 1
        else:
            if sa == 0:
                PP = PP + 1
            else:
                DP = DP + 1

    try:
        human_fairness = (PP) / ((PP)+(PN)) - (DP) / ((DP)+(DN))
    except:
        human_fairness = 0
        
    human_acc = accuracy_score(Y_final, Y[:len(Y_final)])
    
    processed_df = pd.DataFrame.from_dict(list(processed.keys()))
    processed_df.columns = attr_list[:-1]
    data_fairness_matrix = fatf_dfm.systemic_bias(np.array(list(processed_df.to_records(index=False))), np.array(Y_final), protected)
    is_data_unfair = fatf_dfm.systemic_bias_check(data_fairness_matrix)
    unfair_pairs_tuple = np.where(data_fairness_matrix)
    unfair_pairs = []
    for i, j in zip(*unfair_pairs_tuple):
        pair_a, pair_b = (i, j), (j, i)
        if pair_a not in unfair_pairs and pair_b not in unfair_pairs:
            unfair_pairs.append(pair_a)
    if is_data_unfair:
        unfair_n = len(unfair_pairs)
    else:
        unfair_n = 0
        
    return human_fairness, human_acc, unfair_n


def evaluation_frank (X_test, Y_test, model, protected):
    frank_preds = []

    PP, DP, PN, DN = 0, 0, 0, 0

    for x_t, y_t in zip(X_test, Y_test):

        test_pred = model.predict_one(x_t)

        frank_preds.append(test_pred)

        if test_pred == True:
            if x_t[protected[0]] == 0: #0 Male, 1 Female in our tests
                PP = PP + 1
            else:
                DP = DP + 1
        else:
            if x_t[protected[0]] == 0:
                PN = PN + 1
            else:
                DN = DN + 1

    try:
        frank_fairness = (PP) / ((PP)+(PN)) - (DP) / ((DP)+(DN))
    except:
        frank_fairness = 0

    frank_acc = accuracy_score(frank_preds, Y_test)
    
    return frank_fairness, frank_acc

def get_examples(processed, x, model, attr_list, cats, N_BINS, N_VAR, MAX):
    processed_df = pd.DataFrame.from_dict(list(processed.keys()))
    processed_df.columns = attr_list[:-1]

    binned_X = processed_df.copy()
    feats = dict()
    for f in processed_df.columns:
        if f in cats:
            feats[f] = processed_df[f].unique()
        else:
            if len(processed_df[f].unique()) <= N_BINS:
                feats[f] = processed_df[f].unique()
            else:
                binned_X['bins'] = pd.cut(processed_df[f], N_BINS)
                binned_X['median'] = binned_X.groupby('bins')[f].transform('median')
                feats[f] = binned_X['median'].unique()


    all_combinations = []
    for i in range(N_VAR):
        combination = []
        for feat_comb in combinations(feats.keys(), i+1):
            combination.append(feat_comb)
        all_combinations.append(combination)


    ok_feats_against = []
    all_cf_against = []
    ok_feats_pro = []
    all_cf_pro = []
    
    for combination in all_combinations:
        for feat_set in combination:
            if len([f for f in feat_set if f in ok_feats_against]) == 0 and len(all_cf_against) < MAX:
                #print (feat_set)
                list_of_values = []
                for f in feat_set:
                    list_of_values.append(feats[f])
                cf_x = x.copy()

                for val_comb in product(*list_of_values):
                    if len([f for f in feat_set if f in ok_feats_against]) == 0 and len(all_cf_against) < MAX:
                        for val, f in zip(val_comb, feat_set):
                            #idx = list(feats.keys()).index(f)
                            #print(f, idx, "--->", val)
                            cf_x[f] = val
                            #cf_x_model.at[0,f]=val
                        #print(np.array(cf_x))
                        #print("")
                        if model.predict_one(cf_x) == model.predict_one(x) and list(cf_x.values()) != list(x.values()):
                        # == as they are counterfactual AGAINST THE USER'S DECISION
                        # if we want against the machine =!
                        # second condition to avoid having the same record
                            all_cf_against.append(cf_x)
                            for f in feat_set:
                                ok_feats_against.append(f)
            if len([f for f in feat_set if f in ok_feats_pro]) == 0 and len(all_cf_pro) < MAX:
                #print (feat_set)
                list_of_values = []
                for f in feat_set:
                    list_of_values.append(feats[f])
                cf_x = x.copy()

                for val_comb in product(*list_of_values):
                    if len([f for f in feat_set if f in ok_feats_pro]) == 0 and len(all_cf_pro) < MAX:
                        for val, f in zip(val_comb, feat_set): #### controllare se cambiano 2+ feats 
                            #idx = list(feats.keys()).index(f)
                            #print(f, idx, "--->", val)
                            cf_x[f] = val
                            #cf_x_model.at[0,f]=val
                        #print(np.array(cf_x))
                        #print("")
                        if model.predict_one(cf_x) != model.predict_one(x) and list(cf_x.values()) != list(x.values()):
                        # != as they are counterfactual IN FAVOR OF THE USER'S DECISION
                        # if we want against the machine =!
                        # second condition to avoid having the same record
                            all_cf_pro.append(cf_x)
                            for f in feat_set:
                                ok_feats_pro.append(f)


    #print("These records are similar, and should be labelled:", model.predict_one(cf_x))
    #print(attr_list)
    #for e in all_cf:
        #print (list(e.values()))
    return all_cf_pro, all_cf_against