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

#################################################  1. IMPORTING DATA  #################################################

network_email = load_network('/Users/miha/Desktop/Mag./Podatki/EMAIL/email-edges', via_ncol = True, weighted = True, sep = ' ')
network_email_gcc = get_gcc(network_email) # [986, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

network_email_info = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/EMAIL/email-nodes.csv', sep=' ') # open twitter attributes file
network_email_info = network_email_info.rename(columns = {'Node':'twitterId'}) # changing the name for easier use of other code
network_email_info['twitterId'] = network_email_info['twitterId'].apply(str) # twitterid to string

with open("Logs/log_email.txt", "w") as file:
    file.write("Network email loaded \n")

#######################################################################################################################

# check distribution of ground truth group sizes
number_of_groups = len(set(network_email_info.Label))

####################################  2. RUNNING COMMUNITY DETECTION  #################################################

n = 100

# load objects (when already calculated)
#results_email = {}
#with open('/Users/miha/Google Drive/Koda/Python/Objects/email.pkl', 'rb') as f: 
#    results_email['size_table'], results_email['modularity_table'], comparison_table_email, gt_combined = pickle.load(f)

results_email = compare_algorithms(n, network_dict = network_email_gcc, log_file_name='Logs/log_email')
results_email['modularity_table'].to_csv('/Users/miha/Google Drive/Koda/Python/Tables/modularity_email.csv')
results_email['size_table'].to_csv('/Users/miha/Google Drive/Koda/Python/Tables/size_email.csv')

### plot modularity and size distribution
plot_modularity(results_email['modularity_table'], y_name = 'modularity', network_type = 'e-pošta', save=True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim=[0.3,0.8])
plot_size(results_email['size_table'], y_name = 'size', network_type = 'e-pošta', save=True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', y=number_of_groups)

#######################################################################################################################

##########################  3. PAIRWISE COMPARISON OF ALL PARTITIONS  #################################################

## compute pairwise metrics
comparison_table_email = compute_pairwise_metrics(results_email['all_partitions'], network_email_gcc, log_file_name='Logs/log_email')
comparison_table_email.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/pairwise_comparison_email.csv')


## plot metric distribution
plot_metric_comparison('pairwise_comparison','nmi', comparison_table_email, network_type = 'e-pošta', save=True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim = [0, 1.02])
plot_metric_comparison('pairwise_comparison','rand', comparison_table_email, network_type = 'e-pošta', save=True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim = [0, 1.02])
plot_metric_comparison('pairwise_comparison','sj', comparison_table_email, network_type = 'e-pošta', save=True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim = [None, None])
plot_metric_comparison('pairwise_comparison','fs', comparison_table_email, network_type = 'e-pošta', save=True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim = [0,  1.02])

######################################################################################################################

##########################  4. GROUND TRUTH MATCHING  ################################################################

# ground truth matching
gt_matching_louvain = get_gt_matching(results_email['all_partitions'], 'Louvain', network_email_gcc, network_email, network_email_info, 'Label', filter_eu_members = False, filter_gcc = True, network_type='email', log_file_name='Logs/log_email')
gt_matching_dirlouvain = get_gt_matching(results_email['all_partitions'], 'Directed Louvain', network_email_gcc, network_email, network_email_info, 'Label', filter_eu_members = False, filter_gcc = True, network_type='email', log_file_name='Logs/log_email')
gt_matching_leiden = get_gt_matching(results_email['all_partitions'], 'Leiden', network_email_gcc, network_email, network_email_info, 'Label', filter_eu_members = False, filter_gcc = True, network_type='email', log_file_name='Logs/log_email')
gt_matching_infomap = get_gt_matching(results_email['all_partitions'], 'Infomap', network_email_gcc, network_email, network_email_info, 'Label', filter_eu_members = False, filter_gcc = True, network_type='email', log_file_name='Logs/log_email')
gt_matching_oslom = get_gt_matching(results_email['all_partitions'], 'Oslom', network_email_gcc, network_email, network_email_info, 'Label', filter_eu_members = False, filter_gcc = True, network_type='email', log_file_name='Logs/log_email')

# combine and save to csv
gt_combined = pd.concat([add_algorithm_name_column(gt_matching_louvain, 'louvain'), add_algorithm_name_column(gt_matching_dirlouvain, 'directed louvain'), add_algorithm_name_column(gt_matching_leiden, 'leiden'),add_algorithm_name_column(gt_matching_infomap, 'infomap'),add_algorithm_name_column(gt_matching_oslom, 'oslom')])
gt_combined.to_csv('/Users/miha/Google Drive/Koda/Python/Tables/gt_comparison_table_email.csv')

# plot gt matching
plot_metric_comparison('gt_comparison', metric_type='fs', comparison_table=gt_combined, network_type = 'e-pošta', save = True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim = [0, 1.02])
plot_metric_comparison('gt_comparison', metric_type='rand', comparison_table=gt_combined, network_type = 'e-pošta', save = True, show=False, title='Evr. raziskovalna institucija  (omrežje e-pošte)', ylim = [0, 1.02])

with open('Logs/log_email.txt', "a") as f:
        f.write('Finished')


######################################################################################################################
# save objects
with open('Objects/email.pkl', 'wb') as f:
    pickle.dump([results_email['size_table'], results_email['modularity_table'], comparison_table_email, gt_combined], f)
