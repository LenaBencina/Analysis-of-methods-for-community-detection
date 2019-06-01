#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################################################################
### This file contains everything related with plotting average moduluarity.
### Plotting is made with seaborn.
### Different ways of plotting was used  = testing code is included.
##################################################################################################################################################################

"""
Created on Wed Mar 20 09:01:18 2019

@author: Lena
"""

# Libraries
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
from collections import Counter
import matplotlib.pyplot as mplot
mplot.switch_backend("TkAgg")
from importlib import reload
os.chdir("/Users/miha/Google Drive/Koda/Python")
from CD-functions import *

### Main networks
# Retweet network
network_only = load_network('/Users/miha/Desktop/Mag./Podatki/EP/edges_from[oct2014]_to[sep2015]_lang[all]_meps[only]', via_ncol = True, weighted = True, sep = ',')
network_only_info = load_network_info('/Users/miha/Desktop/Mag./Podatki/EP/nodes_from[oct2014]_to[sep2015]_lang[all]_meps[only].csv', all_columns = False, columns = possible_gt) # load network info dataframe 
# load objects (results)
results_only = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/only.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
	results_only['size_table'], results_only['modularity_table'], comparison_table_only, gt_combined_only = pickle.load(f)
# Email network
network_email = load_network('/Users/miha/Desktop/Mag./Podatki/EMAIL/email-edges', via_ncol = True, weighted = True, sep = ' ')
network_email_gcc = get_gcc(network_email) # [986, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
network_email_info = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/EMAIL/email-nodes.csv', sep=' ') # open twitter attributes file
network_email_info = network_email_info.rename(columns = {'Node':'twitterId'}) # changing the name for easier use of other code
network_email_info['twitterId'] = network_email_info['twitterId'].apply(str) # twitterid to string
# load objects
results_email = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/email.pkl', 'rb') as f: 
    results_email['size_table'], results_email['modularity_table'], comparison_table_email, gt_combined = pickle.load(f)
# LFR 1
network_type = '1'
network_lfr1 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info1 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info1['twitterId'] = network_lfr_info1['twitterId'].apply(str) # twitterid to string
number_of_groups1 = len(set(network_lfr_info1.Label))
# load objects
results_lfr1 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr1['size_table'], results_lfr1['modularity_table'], comparison_table_lfr1, gt_combined1 = pickle.load(f)
# LFR 4
network_type = '4'
network_lfr4 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info4 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info4['twitterId'] = network_lfr_info4['twitterId'].apply(str) # twitterid to string
number_of_groups4 = len(set(network_lfr_info4.Label))
# load objects
results_lfr4 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr4['size_table'], results_lfr4['modularity_table'], comparison_table_lfr4, gt_combined4 = pickle.load(f)
# LFR 5
network_type = '5'
network_lfr5 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info5 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info5['twitterId'] = network_lfr_info5['twitterId'].apply(str) # twitterid to string
number_of_groups5 = len(set(network_lfr_info5.Label))
# load objects
results_lfr5 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr5['size_table'], results_lfr5['modularity_table'], comparison_table_lfr5, gt_combined5 = pickle.load(f)
# LFR 7
network_type = '7'
network_lfr7 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info7 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info7['twitterId'] = network_lfr_info7['twitterId'].apply(str) # twitterid to string
number_of_groups7 = len(set(network_lfr_info7.Label))
# load objects
results_lfr7 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr7['size_table'], results_lfr7['modularity_table'], comparison_table_lfr7, gt_combined7 = pickle.load(f)

### Additional LFR networks
# LFR 2
network_type = '2'
network_lfr2 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info2 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info2['twitterId'] = network_lfr_info2['twitterId'].apply(str) # twitterid to string
number_of_groups2 = len(set(network_lfr_info2.Label))
# load objects
results_lfr2 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr2['size_table'], results_lfr2['modularity_table'], comparison_table_lfr2, gt_combined2 = pickle.load(f)
# LFR 3
network_type = '3'
network_lfr3 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info3 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info3['twitterId'] = network_lfr_info3['twitterId'].apply(str) # twitterid to string
number_of_groups3 = len(set(network_lfr_info3.Label))
# load objects
results_lfr3 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr3['size_table'], results_lfr3['modularity_table'], comparison_table_lfr3, gt_combined3 = pickle.load(f)
# LFR 6
network_type = '6'
network_lfr6 = load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network' + network_type, via_ncol = True, weighted = True, sep = ' ')
network_lfr_info6 = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/LFR/community' + network_type + '.csv', sep='\t', header=None, names=['twitterId', 'Label']) # open twitter attributes file
network_lfr_info6['twitterId'] = network_lfr_info6['twitterId'].apply(str) # twitterid to string
number_of_groups6 = len(set(network_lfr_info6.Label))
# load objects
results_lfr6 = {}
with open('/Users/miha/Google Drive/Koda/Python/Objects/lfr' + network_type + '.pkl', 'rb') as f: 
    results_lfr6['size_table'], results_lfr6['modularity_table'], comparison_table_lfr6, gt_combined6 = pickle.load(f)


# Plot modularity alltogether
table_only = results_only['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table_email = results_email['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table1 = results_lfr1['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table4 = results_lfr4['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table5 = results_lfr5['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table7 = results_lfr7['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})

###

table2 = results_lfr2['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table3 = results_lfr3['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
table6 = results_lfr6['modularity_table'][['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})


# try1
#table_only['network'] = 'parlament'
#table_email['network'] = 'email'
#table1['network'] = '1'
#table4['network'] = '4'
#table5['network'] = '5'
#table7['network'] = '7'

#all_lfr = pd.concat([table1, table4, table5, table7])
#table_plot = pd.melt(all_lfr, id_vars = ['network'], var_name = 'method', value_name = 'modularity')
#sns.set_style('whitegrid', {'grid.color': '0.94'})
#sns_plot = sns.pointplot('method', y= 'modularity', hue = 'network', data=table_plot, palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f']))
#sns_plot = sns.catplot(x="network", y="modularity", hue="method", data=table_plot)
#sns_plot.set(ylim=[-0.05, 1])
#sns_plot.set_xticklabels(rotation=30)
#sns_plot.set(xlabel='Metoda', ylabel='Modularnost')
#sns_plot.despine(right=True)
#plt.tight_layout(rect=[0, 0, 0.97, 0.97])
#plt.show()
#plt.clf()



# try2

# main
mean_only = table_only.mean().to_frame().transpose()
mean_email = table_email.mean().to_frame().transpose()
mean1 = table1.mean().to_frame().transpose()
mean4 = table4.mean().to_frame().transpose()
mean5 = table5.mean().to_frame().transpose()
mean7 = table7.mean().to_frame().transpose()
# combine data
mean_all = pd.concat([mean1, mean4, mean5, mean7, mean_only, mean_email])
mean_all['network'] = ['LFR 1','LFR 4','LFR 5','LFR 7', 'Parlament EU', 'Evr. raziskovalna \n institucija']
mean_plot = pd.melt(mean_all, id_vars = ['network'], var_name = 'Metoda', value_name = 'modularity')
# plot
sns.set(font_scale=1.1)
sns.set_style('whitegrid', {'grid.color': '0.94'})
sns_plot = sns.catplot(x = 'network', y = 'modularity', data = mean_plot, hue = 'Metoda', kind = 'point', legend=False, dodge=0.45, join=False, aspect=1.7, palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f', '#7575a3']))
sns_plot.set(ylim=[-0.05, 1])
sns_plot.set_xticklabels(rotation=30)
sns_plot.set(xlabel='Omrežje', ylabel='Povprečna modularnost')
sns.despine(right=True, left=True, bottom=True, top=True)
plt.legend(bbox_to_anchor=(0.99, 1.2), loc=1, ncol=5, borderpad = 0.3, frameon = True, framealpha = 0.3)
plt.tight_layout(rect=[0, 0, 1, 0.9])
#plt.show()
plt.savefig('/Users/miha/Google Drive/Koda/Python/Grafi/modularnost.png', dpi = 500)
plt.clf()


# attachment
mean2 = table2.mean().to_frame().transpose()
mean3 = table3.mean().to_frame().transpose()
mean6 = table6.mean().to_frame().transpose()
# combine data
mean_all = pd.concat([mean2, mean3, mean6])
mean_all['network'] = ['LFR 2','LFR 3','LFR 6']
mean_plot = pd.melt(mean_all, id_vars = ['network'], var_name = 'Metoda', value_name = 'modularity')
# plot
sns.set(font_scale=1.1)
sns.set_style('whitegrid', {'grid.color': '0.94'})
sns_plot = sns.catplot(x = 'network', y = 'modularity', data = mean_plot, hue = 'Metoda', kind = 'point', legend=False, dodge=0.3, join=False, aspect=1.7, palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f', '#7575a3']))
sns_plot.set(ylim=[-0.05, 1])
sns_plot.set_xticklabels(rotation=30)
sns_plot.set(xlabel='Omrežje', ylabel='Povprečna modularnost')
sns.despine(right=True, left=True, bottom=True, top=True)
plt.legend(bbox_to_anchor=(0.99, 1.2), loc=1, ncol=5, borderpad = 0.3, frameon = True, framealpha = 0.3)
plt.tight_layout(rect=[0, 0, 1, 0.9])
#plt.show()
plt.savefig('/Users/miha/Google Drive/Koda/Python/Grafi/modularnost2.png', dpi = 500)
plt.clf()


## all together
mean_all = pd.concat([mean1, mean2, mean3, mean4, mean5, mean6, mean7])#, mean_only, mean_email])
mean_all['network'] = ['LFR 1','LFR 2','LFR 3','LFR 4','LFR 5','LFR 6','LFR 7']# 'Parlament EU', 'Evr. raziskovalna \n institucija']
mean_plot = pd.melt(mean_all, id_vars = ['network'], var_name = 'Metoda', value_name = 'modularity')
# plot
sns.set(font_scale=1.1)
sns.set_style('whitegrid', {'grid.color': '0.94'})
sns_plot = sns.catplot(x = 'network', y = 'modularity', data = mean_plot, hue = 'Metoda', kind = 'point', legend=False, dodge=0.5, join=False, aspect=1.7, palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f', '#7575a3']))
sns_plot.set(ylim=[-0.05, 1])
sns_plot.set_xticklabels(rotation=30)
sns_plot.set(xlabel='Omrežje', ylabel='Povprečna modularnost')
sns.despine(right=True, left=True, bottom=True, top=True)
plt.legend(bbox_to_anchor=(0.99, 1.2), loc=1, ncol=5, borderpad = 0.3, frameon = True, framealpha = 0.3)
plt.tight_layout(rect=[0, 0, 1, 0.9])
#plt.show()
plt.savefig('/Users/miha/Google Drive/Koda/Python/Grafi/modularnost_all.png', dpi = 500)
plt.clf()

