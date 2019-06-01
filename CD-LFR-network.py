#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################################################################
### This file is used for the analysis of Retweet network using functions from CD-functions file.
### The analysis includes: loading network, running community detection n times on it, comparing generated partitions pairwise, comparing generated partitions
### with ground truth partitions, plotting distributions of: modularity values, number of detected communities and other 'metric' values such as
### F-measure (B-cubed), Rand index, Normalized mutual information and Split-join distance, however the code could be modifies for using other 'metrics' as well.
##################################################################################################################################################################

"""
Created on Fri Nov 23 12:02:14 2018

@author: Lena
"""

# Libraries
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import igraph as ig
import louvain
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import networkx as nx
import numpy as np
import leidenalg as la
import matplotlib.pyplot as mplot
mplot.switch_backend("TkAgg")
from importlib import reload
os.chdir("/Users/miha/Google Drive/Koda/Python")
from CD-functions import *
#import CD-functions
#reload(CD-functions)

#######################################################################################################################
# Specify which LFR configuration
import sys
network_type = sys.argv[1]
# network_type = '7'

#################################################  1. IMPORTING DATA  #################################################

network_lfr = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info['twitterId'] = network_lfr_info['twitterId'].apply(str) # twitterid to string

with open('Logs/log_lfr_' + network_type + '.txt', 'w') as file:
    file.write('Network lfr ' + network_type + ' loaded \n')

#######################################################################################################################

# check distribution of ground truth group sizes
number_of_groups = len(set(network_lfr_info.Label))

####################################  2. RUNNING COMMUNITY DETECTION  #################################################

# run community detection
n = 100

# load objects (when already calculated)
#results_lfr = {}
#with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
#    results_lfr['size_table'], results_lfr['modularity_table'], comparison_table_lfr, gt_combined = pickle.load(f)

results_lfr = compare_algorithms(n, network_dict = network_lfr, log_file_name='Logs/log_lfr_' + network_type)
results_lfr['modularity_table'].to_csv('/Users/miha/Google Drive/Koda/Python/Tables/modularity_lfr_' + network_type + '.csv')
results_lfr['size_table'].to_csv('/Users/miha/Google Drive/Koda/Python/Tables/size_lfr_' + network_type + '.csv')


# plot modularity and size distribution # CHANGE Y LIM
plot_modularity(table = results_lfr['modularity_table'], y_name = 'modularity', network_type = network_type, save=True, show=False, title='LFR ' + network_type, ylim = [0,1])
plot_size(table = results_lfr['size_table'], y_name = 'size', network_type = network_type, save=True, show=False, title='LFR ' + network_type, y=number_of_groups)

#######################################################################################################################

##########################  3. PAIRWISE COMPARISON OF ALL PARTITIONS  #################################################

## compute pairwise metrics
comparison_table_lfr = compute_pairwise_metrics(results_lfr['all_partitions'], network_lfr, log_file_name='Logs/log_lfr_' + network_type)
comparison_table_lfr.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/pairwise_comparison_lfr_' + network_type + '.csv')

#comparison_table_lfr.sort_values(by=['method']).to_csv('/Users/miha/Google Drive/Koda/Python/Tables/pairwise_comparison_table_lfr_'+ network_type + '.csv')

## plot metric distribution
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'nmi', comparison_table = comparison_table_lfr, network_type = network_type, save=True, show=False, title='LFR ' + network_type, ylim=[0, 1.02])
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'rand', comparison_table = comparison_table_lfr, network_type = network_type, save=True, show=False, title='LFR ' + network_type, ylim=[0,1.02])
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'sj', comparison_table = comparison_table_lfr, network_type = network_type, save=True, show=False, title='LFR ' + network_type, ylim=[None,None])
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'fs', comparison_table = comparison_table_lfr, network_type = network_type, save=True, show=False, title='LFR ' + network_type, ylim=[0,1.02])

######################################################################################################################

##########################  4. GROUND TRUTH MATCHING  ################################################################

# ground truth matching - full network = all connected
gt_matching_louvain = get_gt_matching(results_lfr['all_partitions'], 'Louvain', network_lfr, network_lfr, network_lfr_info, 'Label', filter_eu_members = False, filter_gcc = False, network_type = 'lfr_' + network_type, log_file_name='Logs/log_lfr_' + network_type)
gt_matching_dirlouvain = get_gt_matching(results_lfr['all_partitions'], 'Directed Louvain', network_lfr, network_lfr, network_lfr_info, 'Label', filter_eu_members = False, filter_gcc = False, network_type = 'lfr_' + network_type, log_file_name='Logs/log_lfr_' + network_type)
gt_matching_leiden = get_gt_matching(results_lfr['all_partitions'], 'Leiden', network_lfr, network_lfr, network_lfr_info, 'Label', filter_eu_members = False, filter_gcc = False, network_type = 'lfr_' + network_type, log_file_name='Logs/log_lfr_' + network_type)
gt_matching_infomap = get_gt_matching(results_lfr['all_partitions'], 'Infomap', network_lfr, network_lfr, network_lfr_info, 'Label', filter_eu_members = False, filter_gcc = False, network_type = 'lfr_' + network_type, log_file_name='Logs/log_lfr_' + network_type)
gt_matching_oslom = get_gt_matching(results_lfr['all_partitions'], 'Oslom', network_lfr, network_lfr, network_lfr_info, 'Label', filter_eu_members = False, filter_gcc = False, network_type = 'lfr_' + network_type, log_file_name='Logs/log_lfr_' + network_type)

# combine and save to csv
gt_combined = pd.concat([add_algorithm_name_column(gt_matching_louvain, 'louvain'), add_algorithm_name_column(gt_matching_dirlouvain, 'directed louvain'), add_algorithm_name_column(gt_matching_leiden, 'leiden'),add_algorithm_name_column(gt_matching_infomap, 'infomap'),add_algorithm_name_column(gt_matching_oslom, 'oslom')])
gt_combined.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/gt_comparison_table_lfr_' + network_type + '.csv')

# plot gt matching
plot_metric_comparison(table_type = 'gt_comparison', metric_type='fs', comparison_table=gt_combined, network_type = network_type, save = True, show=False, title='LFR '+ network_type, ylim = [0,1.02])
plot_metric_comparison(table_type = 'gt_comparison', metric_type='rand', comparison_table=gt_combined, network_type = network_type, save = True, show=False, title='LFR '+ network_type, ylim = [0,1.02])

with open('Logs/log_lfr_' + network_type + '.txt', "a") as f:
        f.write('Finished')

######################################################################################################################
# save objects
with open('Objects/lfr' + network_type + '.pkl', 'wb') as f:
    pickle.dump([results_lfr['size_table'], results_lfr['modularity_table'], comparison_table_lfr, gt_combined], f)

