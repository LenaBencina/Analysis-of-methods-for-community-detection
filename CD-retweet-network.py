#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################################################################
### This file is used for the analysis of Retweet network using functions from CD-functions file.
### The analysis includes: loading network, running community detection n times on it, comparing generated partitions pairwise, comparing generated partitions
### with ground truth partitions, plotting distributions of: modularity values, number of detected communities and other 'metric' values such as
### F-measure (B-cubed), Rand index, Normalized mutual information and Split-join distance, however the code could be modifies for using other 'metrics' as well.
##################################################################################################################################################################

"""
Created on Thu Oct 18 14:53:53 2018

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
os.chdir("/Users/miha/Google Drive/Koda/Python") # directory of CD-functions file
from CD-functions import *
#import CD-functions
#reload(CD-functions)

#################################################  1. IMPORTING DATA  #################################################

### load network with EU members only - EDGES TABLE
network_only = load_network('/Users/miha/Desktop/Mag./Podatki/EP/edges_from[oct2014]_to[sep2015]_lang[all]_meps[only]', via_ncol = True, weighted = True, sep = ',')

# relevant network info - NODES TABLE
network_only_info = load_network_info('/Users/miha/Desktop/Mag./Podatki/EP/nodes_from[oct2014]_to[sep2015]_lang[all]_meps[only].csv', all_columns = False, columns = possible_gt) # load network info dataframe 

# network only is connected - no need of reduction

with open("Logs/log_only.txt", "w") as file:
    file.write("Network only loaded \n")
    
#######################################################################################################################

# check distribution of ground truth group sizes
number_of_groups_only = len(set(network_only_info.mepGroupShort))

####################################  2. RUNNING COMMUNITY DETECTION  #################################################

n = 100 # repetitions

# load objects (when already calculated)
#results_only = {}
#with open('/Users/miha/Google Drive/Koda/Python/Objects/only.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
#    results_only['size_table'], results_only['modularity_table'], comparison_table_only, gt_combined_only = pickle.load(f)

results_only = compare_algorithms(n, network_dict = network_only, log_file_name='Logs/log_only')
results_only['modularity_table'].to_csv('/Users/miha/Google Drive/Koda/Python/Tables/modularity_only.csv')
results_only['size_table'].to_csv('/Users/miha/Google Drive/Koda/Python/Tables/size_only.csv')

### plot modularity and size distribution
plot_modularity(table = results_only['modularity_table'], y_name = 'modularity', network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim=[0.3, 0.8])
plot_size(table = results_only['size_table'], y_name = 'size', network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', y=number_of_groups_only)

#######################################################################################################################

##########################  3. PAIRWISE COMPARISON OF ALL PARTITIONS  #################################################

## compute pairwise metrics
comparison_table_only = compute_pairwise_metrics(results_only['all_partitions'], network_only, log_file_name='Logs/log_only')
comparison_table_only.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/pairwise_comparison_only.csv')

## plot metric distribution
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'nmi', comparison_table = comparison_table_only, network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim = [0, 1.02])
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'rand', comparison_table = comparison_table_only, network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim = [0, 1.02])
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'sj',comparison_table =  comparison_table_only, network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim = [None, None])
plot_metric_comparison(table_type = 'pairwise_comparison', metric_type = 'fs',comparison_table =  comparison_table_only, network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim = [0, 1.02])


######################################################################################################################

##########################  4. GROUND TRUTH MATCHING  ################################################################

gt_group = 'mepGroupShort'

gt_combined_only = pd.DataFrame()
for algorithm in ['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']:
    gt_only_group = get_gt_matching(results_only['all_partitions'], algorithm, network_only, network_only, network_only_info, gt_group, filter_eu_members = False, filter_gcc = False, network_type = 'EU_only', log_file_name='Logs/log_only')
    gt_combined_only = pd.concat([gt_combined_only, add_algorithm_name_column(gt_only_group, algorithm)])

# save
gt_combined_only.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/gt_comparison_only_'+ gt_group + '.csv')

# plot
plot_metric_comparison('gt_comparison', metric_type='fs', comparison_table=gt_combined_only, network_type = 'Parlament EU' , save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim = [0, 1.02])
plot_metric_comparison('gt_comparison', metric_type='rand', comparison_table=gt_combined_only, network_type = 'Parlament EU', save=True, show=False, title='Parlament EU  (omrežje retvitov)', ylim = [0, 1.02])

### additional part: 
# comparing with ground truth as countries - in the final stage political party (gt_group = 'mepGroupShort') was taken as ground truth

#gt_group = 'mepCountry'

#gt_combined_only = pd.DataFrame()
#for algorithm in ['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']:
#    gt_only_group = get_gt_matching(results_only['all_partitions'], algorithm, network_only, network_only, network_only_info, gt_group, filter_eu_members = False, filter_gcc = False, network_type = 'EU_only', log_file_name='Logs/log_only')
#    gt_combined_only = pd.concat([gt_combined_only, add_algorithm_name_column(gt_only_group, algorithm)])
#
#gt_combined_only.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/gt_comparison_only_'+ gt_group + '.csv')
# plot
#plot_metric_comparison('gt_comparison', metric_type='fs', comparison_table=gt_combined_only, network_type = 'only_' + gt_group, save=True, show=False, title='Ground truth comparison of B CUBED for EU (only) - ' + gt_group, ylim = [0, 1])
#plot_metric_comparison('gt_comparison', metric_type='rand', comparison_table=gt_combined_only, network_type = 'only_' + gt_group, save=True, show=False, title='Ground truth comparison of RAND for EU (only) - ' + gt_group, ylim = [0, 1])

with open('Logs/log_only.txt', "a") as f:
        f.write('Finished')

######################################################################################################################
# save objects
with open('Objects/only.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([results_only['size_table'], results_only['modularity_table'], comparison_table_only, gt_combined_only], f)
