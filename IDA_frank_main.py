# Importing required libraries and modules
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
from classes import *  # Importing custom classes
from frank_algo import *  # Importing custom algorithms


class Frank:
    """
    Class representing the Frank decision-making process.

    Attributes:
        RULE (bool): Flag indicating whether predefined rules are applied.
        PAST (bool): Flag indicating whether past decisions are considered.
        SKEPT (bool): Flag indicating whether skepticism is applied.
        GROUP (bool): Flag indicating whether group fairness checks are performed.
        EVA (bool): Flag indicating whether evaluation is performed.
        n_bins (int): Number of bins for XAI (syntethic records).
        n_var (int): Number of variables for XAI (syntethic records).
        maxc (int): Maximum number of modifief features for XAI (syntethic records).
        simulated_user (object): Simulated user model.
        train_check (bool): Flag indicating whether the model is pre-trained.
        Y (list): Target labels of data streams.
        X (list): Feature vectors of data streams.
        X_test (list): For eva.
        Y_test (list): For eva.

        attr_list (list): List of attribute names.
        fairness_records (list): Records for fairness evaluation.
        protected (list): List of protected (sensitive) attributes.
        protected_values (numpy.ndarray): Unique values of protected attributes.
        cats (list): List of categorical attributes.
        rule_att (str): Attribute for predefined rules.
        rule_value (bool): Value for predefined rules.
        model (object): Incremental machine learning model.

    Methods:
        __init__: Initializes the Frank object.
        train: Trains the model.
        start: Starts the decision-making process.
    """

    def __init__(self, user_model, df, target, protected, cats, rule_att, rule_value, RULE, PAST, SKEPT, GROUP, X_test,
                 Y_test, EVA, n_bins, n_var, maxc):
        """
        Initializes the Frank object.

        Args:
            user_model (object): Simulated user model.
            df (DataFrame): Input DataFrame.
            target (str): Target column name.
            protected (list): List of protected attributes.
            cats (list): List of categorical attributes.
            rule_att (str): Attribute for predefined rules.
            rule_value (bool): Value for predefined rules.
            RULE (bool): Flag indicating whether Ideal Rule Check is performed.
            PAST (bool): Flag indicating whether Individual Fairness Check is performed..
            SKEPT (bool): Flag indicating whether Skeptical Learning Check is performed.
            GROUP (bool): Flag indicating whether Group Fairness Check are performed.
            X_test (list): Test feature vectors.
            Y_test (list): Test target labels.
            EVA (bool): Flag indicating whether evaluation is performed.
            EVA (bool): Flag indicating whether evaluation is performed.
            n_bins (int): Number of bins for XAI (syntethic records).
            n_var (int): Number of variables for XAI (syntethic records).

        Returns:
            None
        """

        # Setting attributes
        self.RULE = RULE
        self.PAST = PAST
        self.SKEPT = SKEPT
        self.GROUP = GROUP
        self.EVA = EVA
        self.n_bins = n_bins
        self.n_var = n_var
        self.maxc = maxc
        self.simulated_user = user_model
        self.train_check = False
        self.Y = list(df[target])
        self.Y = [bool(y) for y in self.Y]
        self.X = df.loc[:, df.columns != target]
        self.X = list(self.X.to_dict(orient='index').values())
        self.X_test = X_test
        self.Y_test = Y_test
        self.attr_list = list(df.columns)
        self.fairness_records = [len(self.X) - 1]
        for i in range(0, 100, 5)[1:]:
            self.fairness_records.append(percentage(i, len(self.X)))
        self.protected = protected
        self.protected_values = df[protected[0]].unique()
        self.cats = cats
        self.rule_att = rule_att
        self.rule_value = rule_value
        self.model = tree.ExtremelyFastDecisionTreeClassifier(grace_period=20, nominal_attributes=cats)
        self.processed = dict()
        self.evaluation_results = []
        self.stats = dict()
        self.stats[False] = dict()
        self.stats[True] = dict()
        for e in ['user', 'machine']:
            self.stats[False][e] = dict()
            self.stats[True][e] = dict()
            self.stats[False][e]['tried'] = 0
            self.stats[True][e]['tried'] = 0
            self.stats[False][e]['got'] = 0
            self.stats[True][e]['got'] = 0
            if e == 'user':
                self.stats[False][e]['conf'] = 1
                self.stats[True][e]['conf'] = 1
            else:
                self.stats[False][e]['conf'] = 0
                self.stats[True][e]['conf'] = 0

        #various counters for testing/debugging purposes
        self.rules_count = 0
        self.past_count = 0
        self.ok_count = 0
        self.no_count = 0
        self.xai_ok = 0
        self.xai_no = 0
        self.skept_count = 0
        self.agree_count = 0
        self.disagree_count = 0

    def train(self, X_frank_train, Y_frank_train):
        """
        Pre-training the model.

        Args:
            X_frank_train (list): Training feature vectors.
            Y_frank_train (list): Training target labels.

        Returns:
            None
        """

        self.X_frank_train = X_frank_train
        self.Y_frank_train = Y_frank_train

        for x, y in zip(X_frank_train, Y_frank_train):
            self.model.learn_one(x, y)

        self.train_check = True
        print("Model trained")

    def start(self):
        """
        Starts the Hybrid Decision-Making process.

        Returns:
            processed (dict): Processed records.
            evaluation_results (list): Evaluation results.
        """

        for i in tqdm(range(len(self.X))):

            relabel = False #When this is set to True, Re-Labelling is triggered

            x = self.X[i]
            y = self.Y[i]

            record = tuple(list(x.values()))
            record_user = np.array(list(x.values())).reshape(1, -1)

            user_truth = self.simulated_user.predict(np.array(record).reshape(1, -1)[0], y)
            prediction = self.model.predict_one(x)

            if record in list(self.processed.keys()): #Duplicated record
                self.processed[record]['times'] += 1
                print("Record already processed...")
                old_decision = self.processed[record]['decision']

                if user_truth == old_decision:
                    print("And you are consistent! Decision accepted.")
                    decision = old_decision

                else:
                    print("Inconsistent. You previously said:", old_decision, "Want to change old decision?")

                    confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]

                    if confirm == False:
                        decision = old_decision
                    else:
                        decision = user_truth
                        relabel = True

                self.stats[user_truth]['user']['tried'] += 1
                self.stats[prediction]['machine']['tried'] += 1

                if decision == user_truth:
                    self.stats[user_truth]['user']['got'] += 1
                if decision == prediction:
                    self.stats[prediction]['machine']['got'] += 1

            else:

                self.processed[record] = dict()
                self.processed[record]['notes'] = []
                self.processed[record]['vs'] = None
                self.processed[record]['ideal'] = None
                self.processed[record]['times'] = 1

                try:
                    pred_proba = self.model.predict_proba_one(x)[prediction]
                except:
                    pred_proba = 0

                try:
                    user_proba = self.model.predict_proba_one(x)[user_truth]
                except:
                    print("Still unlearned...")
                    user_proba = 1

                #Skeptical Learning parameters:
                user_confidence = self.stats[user_truth]['user']['conf']
                mach_confidence = self.stats[prediction]['machine']['conf']

                self.stats[user_truth]['user']['tried'] += 1
                self.stats[prediction]['machine']['tried'] += 1

                ideal_value = ideal_record_test(x, self.rule_att, self.rule_value) #Is record covered by Ideal Rule Check?

                vs_records, vs_decision = get_value_swap_records(x, self.processed,
                                                                 self.protected, self.attr_list) #Is record covered by Individual Fairness Check?

                if user_truth == prediction:
                    skepticism = 0
                else:
                    skepticism = mach_confidence * pred_proba - user_confidence * user_proba

                if ideal_value is not None and user_truth != ideal_value and self.RULE: #User is consistent w.r.t. Ideal Rule
                    self.rules_count += 1
                    decision = ideal_value
                    self.processed[record]['ideal'] = False
                    if prediction == ideal_value:
                        self.stats[prediction]['machine']['got'] += 1

                elif ideal_value is not None and user_truth == ideal_value and self.RULE: #User is not consistent w.r.t. Ideal Rule
                    decision = ideal_value
                    self.processed[record]['ideal'] = True
                    if prediction == ideal_value:
                        self.stats[prediction]['machine']['got'] += 1

                elif vs_decision is not None and user_truth != vs_decision and self.PAST: #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                    self.processed[record]['vs'] = True
                    self.past_count += 1
                    for rec in vs_records:
                        self.processed[rec]['vs'] = True
                    confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]
                    if confirm in [0, "0", False]:
                        decision = vs_decision
                        if prediction == vs_decision:
                            self.stats[prediction]['machine']['got'] += 1
                    elif confirm in [1, "1", True]:
                        decision = user_truth
                        self.stats[user_truth]['user']['got'] += 1
                        if prediction == user_truth:
                            self.stats[prediction]['machine']['got'] += 1
                        for rec in vs_records:
                            self.processed[rec]['decision'] = user_truth
                        relabel = True

                elif vs_decision is not None and user_truth == vs_decision and self.PAST: #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                    self.processed[record]['vs'] = True
                    for rec in vs_records:
                        self.processed[rec]['vs'] = True
                    decision = vs_decision
                    if prediction == vs_decision:
                        self.stats[prediction]['machine']['got'] += 1

                else: #Other conditions not triggered. Skeptical Learning Check
                    if user_truth != prediction and self.SKEPT:
                        if skepticism > 0.05:
                            self.skept_count += 1
                            confirm = self.simulated_user.believe()
                            if confirm == None: #The user is provided explanation of the model's decision. Our tests only focused on sythethic records
                                xai_check = 0
                                ex_pro, ex_against = get_examples(self.processed, x, self.model, self.attr_list,
                                                                   self.cats, self.n_bins, self.n_var, self.maxc)
                                for e in ex_pro: #counters for debugging purposes
                                    user_opinion = self.simulated_user.predict(
                                        np.array(list(e.values())).reshape(1, -1)[0], None)
                                    if user_opinion == prediction:
                                        xai_check += 1
                                        self.xai_ok += 1
                                    else:
                                        self.xai_no += 1
                                for e in ex_against:
                                    user_opinion = self.simulated_user.predict(
                                        np.array(list(e.values())).reshape(1, -1)[0], None)
                                    if user_opinion != prediction:
                                        xai_check += 1
                                        self.xai_ok += 1
                                    else:
                                        self.xai_no += 1
                                if (xai_check / len(ex_pro + ex_against)) > 0.5:
                                    confirm = True
                                else:
                                    confirm = False
                            if confirm in [0, "0", False]:
                                self.no_count += 1
                                decision = user_truth
                                self.stats[user_truth]['user']['got'] += 1
                            else:
                                self.ok_count += 1
                                decision = prediction
                                self.stats[prediction]['machine']['got'] += 1
                        else:
                            self.disagree_count += 1
                            decision = user_truth
                            self.stats[user_truth]['user']['got'] += 1
                    else:
                        self.agree_count += 1
                        decision = user_truth
                        self.stats[user_truth]['machine']['got'] += 1
                        self.stats[user_truth]['user']['got'] += 1

                #Once the final decision has been taken, the model is updated. Internal data structure is also updated
                self.processed[record]['dict_form'] = x
                self.processed[record]['decision'] = decision
                self.processed[record]['user'] = user_truth
                self.processed[record]['machine'] = prediction

                self.model.learn_one(x, decision)

            try:
                self.stats[user_truth]['user']['conf'] = self.stats[user_truth]['user']['got'] / self.stats[user_truth]['user']['tried']
            except:
                self.stats[user_truth]['user']['conf'] = 1

            try:
                self.stats[prediction]['machine']['conf'] = self.stats[prediction]['machine']['got'] / self.stats[prediction]['machine']['tried']
            except:
                self.stats[prediction]['machine']['conf'] = 0

            if relabel == True:
                self.model = tree.ExtremelyFastDecisionTreeClassifier(grace_period=20, nominal_attributes=self.cats)

                for x_train_sample, y_train_sample in zip(self.X_frank_train, self.Y_frank_train):
                    self.model.learn_one(x_train_sample, y_train_sample)

                for proc in self.processed.keys():
                    x_relabel = self.processed[proc]['dict_form']
                    y_relabel = self.processed[proc]['decision']
                    self.model.learn_one(x_relabel, y_relabel)

            if i in self.fairness_records and self.GROUP:
                DN, PP, ext = get_fairness(self.model, self.protected, self.processed, self.protected_values)
                fairnes_relabel = DN[:round(len(DN) * 0.25)] + PP[:round(len(PP) * 0.25)]
                for e in fairnes_relabel:
                    self.processed[e[1]]['decision'] = not self.processed[e[1]]['decision']
                self.model = tree.ExtremelyFastDecisionTreeClassifier(grace_period=20, nominal_attributes=self.cats)
                for x_train_sample, y_train_sample in zip(self.X_frank_train, self.Y_frank_train):
                    self.model.learn_one(x_train_sample, y_train_sample)
                for proc in self.processed.keys():
                    x_relabel = self.processed[proc]['dict_form']
                    y_relabel = self.processed[proc]['decision']
                    self.model.learn_one(x_relabel, y_relabel)

            if self.EVA:
                human_fairness, human_acc, systemic = evaluation_human(self.processed, self.protected, self.Y,
                                                                       self.attr_list)
                frank_fairness, frank_acc = evaluation_frank(self.X_test, self.Y_test, self.model, self.protected)
                self.evaluation_results.append([human_fairness, human_acc, systemic, frank_fairness, frank_acc,
                                                self.rules_count, self.past_count,
                                                self.ok_count, self.no_count,
                                                self.xai_ok, self.xai_no,
                                                self.skept_count, self.agree_count, self.disagree_count
                                                ])

        return self.processed, self.evaluation_results
