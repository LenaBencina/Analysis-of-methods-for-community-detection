#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################################################################
### This file contains most of the functions used for empirical comparison of the following community detection methods:
### Louvain (undirected), Louvain (directed), Leiden (directed), Infomap (directed), OSLOM (directed)
### search for 'Users' to find all the directories that needs to be changed
###
### OSLOM and Python OSLOM runner are required for running community detection with OSLOM (run_oslom function) 
### http://www.oslom.org/software.htm
### https://github.com/hhromic/python-oslom-runner
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
import bcubed
from argparse import Namespace
import oslom
from oslom.runner import DEF_OSLOM_EXEC, DEF_OSLOM_ARGS
import time
from scipy.sparse import coo_matrix
from itertools import product

###################################################################################################################################################################
# Global vars
global possible_gt, network_size_dict, to_roman
possible_gt = ['mepGroupShort', 'mepCountry', 'twitterLanguage'] # selecting relevant columns
network_size_dict = {'Parlament EU':460, 'e-pošta':1005, 'LFR 1':1000, 'LFR 2':1000, 'LFR 3':5000, 'LFR 4':5000, 'LFR 5':1000, 'LFR 6':1000, 'LFR 7':1000}
to_roman = {'1':'I', '2':'II', '3':'III', '4':'IV', '5':'V', '6':'VI', '7':'VII'} # # dictionary for converting integer to roman numerals

###################################################################################################################################################################

# Function for converting igraph object to networkx object
def ig_to_nx(graph, directed):
    A = graph.get_edgelist()
    if directed:
        return nx.DiGraph(A) # In case the graph is directed
    else:
        return nx.Graph(A) # In case the graph is undirected

# Function for drawing networkx object with labels
def draw_nx_with_labels(g, labels, font_size, node_size):
    pos=nx.spring_layout(g) # getting the position of nodes
    nx.draw_networkx_nodes(g, pos, node_size=node_size)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels ,font_size=font_size)

# Function for converting pandas dataframe (form: src dest weight) network to igraph object
def pandas_to_igraph(network_df, weighted):
    network_df['Src'] = network_df['Src'].apply(str)
    network_df['Dest'] = network_df['Dest'].apply(str)
    G = ig.Graph(directed = True) # create directed graph object
    vertices = pd.unique(network_df[['Src', 'Dest']].values.ravel('K'))
    G.add_vertices(vertices) # add vertices
    weights = []
    for index, row in network_df.iterrows(): # for each edge in the network
        edge = (row['Src'], row['Dest'])        
        G.add_edges([edge])
        if weighted:
            weights.append(row['Weight'])
    # add weights 
    if weighted:
        G.es['weight'] = weights
    return G

# Function for running Louvain alg on weighted network
def run_louvain(network):
    partition = louvain.find_partition(graph=network, 
                                       partition_type=louvain.ModularityVertexPartition, 
                                       weights=network.es['weight'])
    return partition

# Function for running Leiden alg
def run_leiden(network, i):
    partition = la.find_partition(graph=network,
                                partition_type=la.ModularityVertexPartition,
                                weights =network.es['weight'],
                                seed=i) 
    return partition

# Function for running OSLOM alg (C++ implementation)
def run_oslom(network_oslom, i):
        args = Namespace()
        args.min_cluster_size = 0
        args.oslom_exec = '/Users/miha/Downloads/OSLOM2/oslom_dir'
        args.oslom_args = DEF_OSLOM_ARGS + ["-seed", str(i+1)]
        clusters = oslom.run_in_memory(args, network_oslom)
        return clusters

# Function for saving partition as pickle object
def save_object(var, file_name):
    os.chdir('/Users/miha/Google Drive/Koda/Python/Objekti/')
    with open(file_name, "wb") as f: # save result
        pickle.dump(var, f)
  
# Function for loading partition from pickle object
def load_object(file_name):
    with open(file_name, "rb") as f: # Unpickling
        return pickle.load(f)

# Function for importing network from file path either from ncol file or via csv to igraph
def load_network(file_path, via_ncol, weighted, sep):
    if via_ncol:
        network = ig.Graph.Read_Ncol(f = file_path + '.txt',
                                     names = True,
                                     weights = weighted,
                                     directed=True)
    else:
        network_df = pd.read_csv(file_path + '.csv', sep=sep)
        cols = ['Src', 'Dest']
        if weighted:
            cols.append('Weight')
        network_df.columns = cols
        network = pandas_to_igraph(network_df, weighted) # convert pandas df network to igraph
    # for oslom - tuple network
    network_df = pd.read_csv(file_path + '.csv', sep=sep)
    network_tuples = [tuple(x) for x in network_df.values]
    network_tuples = [(int(t[0]), int(t[1]), t[2]) for t in network_tuples]
    return {'igraph':network, 'tuple': network_tuples}

# Function for importing relevant info (specifying columns) into pandas df
def load_network_info(file_path, all_columns, columns):
    # get info about nodes
    nodes_info = pd.read_csv(file_path, sep=',') # open twitter attributes file
    nodes_info['twitterId'] = nodes_info['twitterId'].apply(str) # twitterid to string
    if all_columns:
        columns = list(nodes_info.columns)
    relevant_info = nodes_info.filter(items=['twitterId'] + columns)
    return relevant_info
    
# Function for converting an undirected network to directed network (IGRAPH!)
def directed_to_undirected(network):
    network_copy = network.copy()
    network_copy.to_undirected(mode="collapse", combine_edges=sum)
    return network_copy

# Function for comparing algorithms n times
def compare_algorithms(n, network_dict, log_file_name):
    network, network_oslom = network_dict['igraph'], network_dict['tuple']
    all_partitions = {'Louvain':[], 'Directed Louvain':[], 'Leiden':[], 'Infomap':[], 'Oslom':[]}
    modularity_table = pd.DataFrame()
    size_table = pd.DataFrame()
    for i in range(0, n): # run alg n times
        louvain.set_rng_seed(i)
        #
        start = time.time() 
        ### 1) directed Louvain
        partition_dl = run_louvain(network)
        all_partitions['Directed Louvain'].append(partition_dl)
        modularity_table.at[i, 'Directed Louvain'] = partition_dl.quality()
        size_table.at[i, 'Directed Louvain'] = len(partition_dl)
        #
        end = time.time()
        with open(log_file_name + ".txt", "a") as f:
                f.write('CD - dir_louvain -: ' + str(i) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
        #
        start = time.time()
        ### 2) directed Leiden
        partition_lei = run_leiden(network, i)
        all_partitions['Leiden'].append(partition_lei)
        modularity_table.at[i, 'Leiden'] = partition_lei.quality()
        size_table.at[i, 'Leiden'] = len(partition_lei)
        #
        end = time.time()
        with open(log_file_name + ".txt", "a") as f:
                f.write('CD - dir_leiden -: ' + str(i) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
        #
        start = time.time()
        ### 3) undirected Louvain
        # create an undirected netowork for comparison
        network_ud = directed_to_undirected(network)
        partition_l = run_louvain(network_ud)
        all_partitions['Louvain'].append(partition_l)
        modularity_table.at[i, 'Louvain'] = partition_l.quality()
        size_table.at[i, 'Louvain'] = len(partition_l)
        #
        end = time.time()
        with open(log_file_name + ".txt", "a") as f:
                f.write('CD - undir_louvain -: ' + str(i) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
        #
        start = time.time()
        ### 4) directed infomap
        partition_i = network.community_infomap(edge_weights = network.es['weight'], trials=1)
        all_partitions['Infomap'].append(partition_i)
        size_table.at[i, 'Infomap'] = len(set(partition_i.membership))
        # modularity
        community_dict_infomap = get_community_dict(partition_i, network, filter_eu_members = False)['mod_dict']
        modularity_table.at[i, 'Infomap'] = get_modularity(network, community_dict_infomap)
        #
        end = time.time()
        with open(log_file_name + ".txt", "a") as f:
                f.write('CD - infomap -: ' + str(i) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
        #
        start = time.time()
        ### 5) directed oslom
        clusters = run_oslom(network_oslom, i)
        all_partitions['Oslom'].append(clusters[0])
        size_table.at[i, 'Oslom'] = clusters[0]['num_found'] # number of clusters found
        # modularity
        community_dict_oslom = get_community_dict_oslom(clusters[0], network, filter_eu_members = False)['mod_dict']
        modularity_table.at[i, 'Oslom'] = get_modularity(network, community_dict_oslom)
        #
        end = time.time()
        with open(log_file_name + ".txt", "a") as f:
                f.write('CD - oslom -: ' + str(i) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
        #
    return {'size_table':size_table, 'modularity_table':modularity_table, 'all_partitions':all_partitions}

# Function for plotting modularity - this function is not up to date with all the plot settings
def plot_modularity(table, y_name, network_type, save, show, title, ylim):
        if 'LFR' in title: # quick fix for getting roman numbers to title
                #title = 'LFR ' + to_roman[network_type]
                network_type = 'LFR ' + network_type
        table = table[['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
        table_plot = pd.melt(table, var_name = 'method', value_name = y_name)
        sns.set_style("whitegrid", {'grid.color': '0.94'})
        sns.set(font_scale=1.1)
        sns_plot = sns.catplot(x = 'method', y = y_name, kind = 'swarm', data = table_plot, palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f', '#7575a3']))
        sns_plot.set(ylim=ylim)
        sns_plot.set_xticklabels(rotation=30)
        sns_plot.set(xlabel='Metoda', ylabel='Modularnost')
        sns_plot.despine(left=True, bottom=True)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.title(title, y = 1.04)
        if save:
                plt.savefig('/Users/miha/Google Drive/Koda/Python/Grafi/' + y_name + '_' + network_type + '.png', dpi = 500)
        if show:
                plt.show()
        plt.clf()

# Function for plotting distribution of number of detected communities
def plot_size(table, y_name, network_type, save, show, title, y):
        if 'LFR' in title: network_type = 'LFR ' + network_type
        table = table[['Louvain', 'Directed Louvain', 'Leiden', 'Infomap', 'Oslom']].rename(index=str, columns={'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'})
        table_plot = pd.melt(table, var_name = 'method', value_name = y_name)    
        # CUSTOM 7b: delete oslom because of the high values
        # table_plot = table_plot[table_plot.method != 'OSLOM']
        # breaks = range(0,60,10)
        # y_lim = [-2,52]
        table_plot[['size']] = table_plot[['size']].astype(int)
        sns.set(font_scale=1.3)
        sns.set_style("whitegrid", {'grid.color': '0.94'})
        sns_plot = sns.catplot(x='method', y="size", kind= 'swarm', aspect=1.2, data=table_plot, palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f', '#7575a3']))
        # set breaks
        breaks_dict = {'Parlament EU':range(5,60,10), 'e-pošta':range(0,55,10), 'LFR 1':range(0,55,10), 'LFR 2':range(0,55,10), 'LFR 3':range(95,185,15), 'LFR 4':range(60, 300, 40), 'LFR 5':range(0, 12, 2), 'LFR 6':range(0, 12, 2), 'LFR 7':range(0, 1200, 200)}
        breaks = breaks_dict[network_type]
        sns_plot.set(yticks = breaks)
        # set ylim
        y_lim_dict = {'Parlament EU':[3,57], 'e-pošta':[-2,52], 'LFR 1':[-2,52], 'LFR 2':[-2,52], 'LFR 3':[92,173], 'LFR 4':[52,268], 'LFR 5':[-0.4,10.4], 'LFR 6':[-0.4,10.4], 'LFR 7':[-40,1040]}
        y_lim = y_lim_dict[network_type]
        sns_plot.set(ylim = y_lim)
        sns_plot.set_xticklabels(rotation=30)
        sns_plot.set(xlabel='Metoda', ylabel='Število skupnosti')
        sns_plot.despine(left=True, bottom=True)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.title(title, y = 1.04)
        if y != None:
                plt.axhline(y=y, c='black', linestyle='--', linewidth=0.9)
        if save:
                plt.savefig('/Users/miha/Google Drive/Koda/Python/Grafi/' + y_name + '_' + network_type + '.png', dpi = 500)
        if show:
                plt.show()
        plt.clf()


# Function for computing pairwise metrics such as variation of info., normalized mutual information, (adjusted) rand index and split-join
def compute_pairwise_metrics(all_partitions, network_dict, log_file_name):
        network = network_dict['igraph']
        n = len(all_partitions['Louvain'])
        ### get community dictionaries for calculating B cubed - F score
        community_dicts_louvain = get_all_community_dicts(all_partitions['Louvain'], network, filter_eu_members=False)['dict']
        community_dicts_louvain_dir = get_all_community_dicts(all_partitions['Directed Louvain'], network, filter_eu_members=False)['dict']
        community_dicts_leiden = get_all_community_dicts(all_partitions['Leiden'], network, filter_eu_members=False)['dict']
        community_dicts_infomap = get_all_community_dicts(all_partitions['Infomap'], network, filter_eu_members=False)['dict']
        # get oslom dicts and lists
        oslom_dicts_and_lists = get_all_community_dicts_oslom(all_partitions['Oslom'], network, filter_eu_members=False)
        #### compute pairwise metrics
        comparison_table = pd.DataFrame(columns=['nmi', 'rand', 'sj', 'fs', 'method'])
        index = 0
        for i in range(0, n): # comparing partition pairs
                for j in range(i+1, n):
                        start = time.time()
                        #
                        # 1) Louvain
                        nmi = (ig.compare_communities(all_partitions['Louvain'][i], all_partitions['Louvain'][j], method = 'nmi', remove_none = False))
                        rand = (ig.compare_communities(all_partitions['Louvain'][i], all_partitions['Louvain'][j], method = 'rand', remove_none = False))
                        sj = (ig.compare_communities(all_partitions['Louvain'][i], all_partitions['Louvain'][j], method = 'split-join', remove_none = False))
                        fs = f_score(community_dicts_louvain[i], community_dicts_louvain[j])
                        comparison_table.loc[index] = [nmi, rand, sj, fs, 'Louvain']
                        index = index + 1
                        #
                        # 2) Directed Louvain
                        nmi = (ig.compare_communities(all_partitions['Directed Louvain'][i], all_partitions['Directed Louvain'][j], method = 'nmi', remove_none = False))
                        rand = (ig.compare_communities(all_partitions['Directed Louvain'][i], all_partitions['Directed Louvain'][j], method = 'rand', remove_none = False))
                        sj = (ig.compare_communities(all_partitions['Directed Louvain'][i], all_partitions['Directed Louvain'][j], method = 'split-join', remove_none = False))
                        fs = f_score(community_dicts_louvain_dir[i], community_dicts_louvain_dir[j])
                        comparison_table.loc[index] = [nmi, rand, sj, fs, 'Directed Louvain']
                        index = index + 1
                        #
                        # 3) Leiden        
                        nmi = (ig.compare_communities(all_partitions['Leiden'][i], all_partitions['Leiden'][j], method = 'nmi', remove_none = False))
                        rand = (ig.compare_communities(all_partitions['Leiden'][i], all_partitions['Leiden'][j], method = 'rand', remove_none = False))
                        sj = (ig.compare_communities(all_partitions['Leiden'][i], all_partitions['Leiden'][j], method = 'split-join', remove_none = False))
                        fs = f_score(community_dicts_leiden[i], community_dicts_leiden[j])
                        comparison_table.loc[index] = [nmi, rand, sj, fs, 'Leiden']
                        index = index + 1
                        #
                        # 4) Infomap
                        nmi = (ig.compare_communities(all_partitions['Infomap'][i], all_partitions['Infomap'][j], method = 'nmi', remove_none = False))
                        rand = (ig.compare_communities(all_partitions['Infomap'][i], all_partitions['Infomap'][j], method = 'rand', remove_none = False))
                        sj = (ig.compare_communities(all_partitions['Infomap'][i], all_partitions['Infomap'][j], method = 'split-join', remove_none = False))
                        fs = f_score(community_dicts_infomap[i], community_dicts_infomap[j])
                        comparison_table.loc[index] = [nmi, rand, sj, fs, 'Infomap']
                        index = index + 1
                        #
                        # 5) Oslom
                        nmi = (ig.compare_communities(oslom_dicts_and_lists[i]['list'], oslom_dicts_and_lists[j]['list'], method = 'nmi', remove_none = False))
                        rand = (ig.compare_communities(oslom_dicts_and_lists[i]['list'], oslom_dicts_and_lists[j]['list'], method = 'rand', remove_none = False))
                        sj = (ig.compare_communities(oslom_dicts_and_lists[i]['list'], oslom_dicts_and_lists[j]['list'], method = 'split-join', remove_none = False))
                        fs = f_score(oslom_dicts_and_lists[i]['dict'], oslom_dicts_and_lists[j]['dict'])
                        comparison_table.loc[index] = [nmi, rand, sj, fs, 'Oslom']
                        index = index + 1
                        #
                        end = time.time()
                        with open(log_file_name + ".txt", "a") as f:
                                f.write('PC: ' + str(i) + '-' + str(j) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
                        #
        return comparison_table


# Function for plotting metric distribution for different algorithms
def plot_metric_comparison(table_type, metric_type, comparison_table, network_type, save, show, title, ylim):
        if 'LFR' in title: # quick fix for getting roman numbers to title
                #title = 'LFR ' + to_roman[network_type]
                network_type = 'LFR ' + network_type
        metric_translation = {'nmi': 'NMI', 'rand': 'Randov indeks', 'sj': 'Razdalja razdeli-združi', 'fs': 'F-mera (B-cubed)'}
        comparison_table.replace({'Directed Louvain': 'Louvain (u)', 'Louvain': 'Louvain (n)', 'Oslom':'OSLOM'}, inplace = True)
        comparison_table.replace({'louvain': 'Louvain (n)', 'directed louvain': 'Louvain (u)', 'leiden': 'Leiden', 'infomap': 'Infomap', 'oslom': 'OSLOM'}, inplace = True)
        ### boxplot
        #sns_plot = sns.boxplot(x = 'method', y = metric_type, data = comparison_table)
        #sns_plot.set_ylim(ylim)
        ### catplot (swarm/violin)
        sns.set(font_scale=1.3)
        sns.set_style("whitegrid", {'grid.color': '0.94'})
        sns_plot = sns.catplot(x = 'method', y = metric_type, data = comparison_table,line_width=0.6, aspect=1.2, kind= 'violin', palette = sns.color_palette(['#F08080', '#6CA6CD', '#8FBC8F','#ffde6f', '#7575a3']), cut = 0)
        # Split join
        if metric_type == 'sj':
                #ylim=[0, 2*network_size_dict[network_type]]
                sj_dict = {'Parlament EU': range(0,990,180), 'e-pošta': range(0,2400,400), 'LFR 1': range(0,2400,400), 'LFR 2': range(0,2400,400), 'LFR 3': range(0,12000,2000), 'LFR 4': range(0,12000,2000), 'LFR 5': range(0,2400,400), 'LFR 6': range(0,2400,400), 'LFR 7': range(0,2400,400)}
                breaks = sj_dict[network_type]
                sns_plot.set(yticks=breaks)
                y_lim_dict = {'Parlament EU': [-36, 936], 'e-pošta': [-80, 2080], 'LFR 1': [-80, 2080], 'LFR 2': [-80, 2080], 'LFR 3': [-400, 10400], 'LFR 4': [-400, 10400], 'LFR 5': [-80, 2080], 'LFR 6': [-80, 2080], 'LFR 7': [-80, 2080]}
                y_lim = y_lim_dict[network_type]
        else: y_lim = [-0.04, 1.04]
        sns_plot.set(ylim=y_lim)
        sns_plot.set_xticklabels(rotation=30)
        sns_plot.set(xlabel = 'Metoda', ylabel = metric_translation[metric_type]) # POPRAVI
        sns_plot.despine(left=True, bottom=True)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.title(title, y = 1.04)
        if save:
                plt.savefig('/Users/miha/Google Drive/Koda/Python/Grafi/' + table_type + '_' + metric_type  + '_' + network_type + '.png', dpi=500)
        if show:
                plt.show()
        plt.clf()

###############################################################################################################################################
# some additional global vars - quick fix

global ext_members_ids
network_ext_info = load_network_info('/Users/miha/Desktop/Mag./Podatki/EP/nodes_from[oct2014]_to[sep2015]_lang[all]_meps[ext].csv',
                                 all_columns = False,
                                 columns = possible_gt) # load network info dataframe 
ext_members_ids = network_ext_info[network_ext_info['mepGroupShort'].notnull()].twitterId.tolist()

#################################################### B-CUBED ##############################################################################
# These functions are needed for conversion of partitions into structures that are needed for calculating B-cubed (F-measure)

def get_community_dict(partition, network, filter_eu_members): # get community dict for each partition for calculating f-score
        if filter_eu_members:
            id_names_dict = {v.index:v['name'] for v in network.vs() if v['name'] in ext_members_ids} # {nodeID: nodeName = 1:109820983}
        else:
            id_names_dict = {v.index:v['name'] for v in network.vs()} # {nodeID: nodeName = 1:109820983}
        #
        community_dict = {}
        community_dict_mod = {}
        community_list = []
        # for each vertexID (0,...,459) & vertexName = twitterID (ex. 192879871)
        for vertex_id, vertex_name in id_names_dict.items():
                community = partition.membership[vertex_id] # get communityId (element index in membership vector = node id)
                community_dict[vertex_name] = set([community]) # {nodeName : {communityId}}
                community_list.append(community)
                community_dict_mod[vertex_id] = community 
        return {'dict':community_dict, 'list':community_list, 'mod_dict':community_dict_mod}


def get_gt_dict(network_info, gt_group, filter_gcc, network_full): # get ground truth dict for calculating f-score
        if filter_gcc:
            ids_to_delete = get_ids_to_delete(network_full, names_output=True)
            network_info = network_info[~network_info.twitterId.isin(ids_to_delete)] 
        gt = network_info[gt_group].values.tolist() # ground-truth=actual group list
        to_set = lambda x: set([x]) # lambda mapping set on each element
        gt_dict = dict(zip(network_info['twitterId'], list(map(to_set, gt)))) # {twitterId: {gt group} = 34085098: {ALDE}}
        return gt_dict

def f_score(community_dict, gt_dict): # calculating f_score
        precision = bcubed.precision(community_dict, gt_dict)
        recall = bcubed.recall(community_dict, gt_dict)
        fscore = bcubed.fscore(precision, recall)
        return fscore

def get_all_community_dicts(all_partitions_one_method, network, filter_eu_members): # 
        community_dicts = []
        community_lists = [] # adding list structure for rand calculations
        for i in range(0, len(all_partitions_one_method)):
            tmp = get_community_dict(all_partitions_one_method[i], network, filter_eu_members)
            community_dicts.append(tmp['dict'])
            community_lists.append(tmp['list'])
        return {'dict':community_dicts,'list':community_lists}


####################### SPECIAL OSLOM FUNCTIONS for B-CUBED ###################################
# These functions are needed for conversion of OSLOM output partitions for calculating B-cubed.

def get_all_community_dicts_oslom(all_partitions_oslom, network, filter_eu_members): # network here is igraph - for getting ids in correct order
        community_dicts = []
        for i in range(0, len(all_partitions_oslom)):
            community_dicts.append(get_community_dict_oslom(all_partitions_oslom[i], network, filter_eu_members))
        return community_dicts

def get_community_dict_oslom(all_oslom_outputs, network, filter_eu_members):
        # dict
        community_dict = {} # for calculating B-cubed 
        community_dict_mod = {} # for calculating modularity
        for cluster in all_oslom_outputs['clusters']:
                for node in cluster['nodes']:
                        node_index = network.vs.find(name=str(node['id'])).index # get index of vertex
                        if filter_eu_members:
                                if str(node['id']) not in ext_members_ids:
                                        #print(node['id'])
                                        continue
                                community_dict[str(node['id'])] = {cluster['id']}
                                community_dict_mod[node_index] = cluster['id']
                        else:
                                community_dict[str(node['id'])] = {cluster['id']}
                                community_dict_mod[node_index] = cluster['id']
        # list
        id_names_list_in_order = list(network.vs()['name']) # twitter ids in order
        index_map = {v: i for i, v in enumerate(id_names_list_in_order)} # get dict {twitter_id: enumeration id}
        sorted_community_dict = sorted(community_dict.items(), key=lambda pair: index_map[pair[0]]) # list of tuples
        #sorted_community_dict = sorted(community_dict.items(), key=lambda pair: index_map[str(int(float(pair[0])))]) # list of tuples
        community_list = [next(iter(pair[1])) for pair in sorted_community_dict] # to dict
        return {'dict':community_dict,'list':community_list, 'mod_dict':community_dict_mod}

######################################################## GROUND TRUTH MATCHING ###################################################################################
# These functions are needed for comparing generated partitions with "ground truth" partitions.

def filter_ext_members(network_info): # filtering network info table (nodes table) by ids with mepGroupShort values
        filtered = network_info[network_info['twitterId'].isin(ext_members_ids)]
        return filtered

def extract_from_set(community_set):
        return next(iter(community_set))

def get_gt_int_partition(gt_list): # map partition of actual groups (parties or countries) to integer list
        d = dict([(y,x+1) for x,y in enumerate(sorted(set(gt_list)))])
        return [d[x] for x in gt_list]

# Main function for ground truth matching
def get_gt_matching(all_partitions, algorithm, network_dict, network_full_dict, network_info, gt_type, filter_eu_members, filter_gcc, network_type, log_file_name): # computin F score and Rand for (partition, ground truth group)
        partitions = all_partitions[algorithm] # get all partitions for specific algorithm
        # get dictionaries for calculating F score
        network, network_full = network_dict['igraph'], network_full_dict['igraph']
        if algorithm=='Oslom':
                community_dicts_and_lists = get_all_community_dicts_oslom(partitions, network, filter_eu_members)
        else:
                community_tmp = get_all_community_dicts(partitions, network, filter_eu_members) # get dictionaries for community membership
                community_dicts, community_lists = community_tmp['dict'], community_tmp['list']
        gt_dict = get_gt_dict(network_info, gt_type, filter_gcc, network_full) # network full = network with all components
        # get ground truth list in correct order for rand index
        id_names_list_in_order = list(network.vs()['name']) # list of twitter ids in correct order
        index_map = {v: i for i, v in enumerate(id_names_list_in_order)}
        sorted_gt_dict = sorted(gt_dict.items(), key=lambda pair: index_map[pair[0]])
        gt_list = [next(iter(pair[1])) for pair in sorted_gt_dict]
        gt_int_partition = get_gt_int_partition(gt_list) # get ground truth partition in integer form
        # calculate metrics
        metrics_table = pd.DataFrame(columns=['fs', 'rand'])
        for i in range(0, len(partitions)): # for each partition
                start = time.time()
                #
                if algorithm == 'Oslom':
                        fs = f_score(community_dicts_and_lists[i]['dict'], gt_dict)
                        rand = ig.compare_communities(community_dicts_and_lists[i]['list'], gt_int_partition, method = 'rand', remove_none = False)
                else:
                        fs = f_score(community_dicts[i], gt_dict)
                        rand = ig.compare_communities(community_lists[i], gt_int_partition, method = 'rand', remove_none = False)
                metrics_table.loc[i] = [fs, rand]
                #
                end = time.time()
                with open(log_file_name + ".txt", "a") as f:
                        f.write('GTC: ' + str(i) + '  TIME: ' + str(round((end-start)/60,4)) + '\n')
        return metrics_table

# Function for adding method names for saving
def add_algorithm_name_column(df, alg_name):
        n = df.shape[0] # number of rows
        df['method'] = n * [alg_name]
        return df

######################################################## GIANT CONNECTED COMPONENT ################################################################################

# Function for identifying ids not included in gcc
def get_ids_to_delete(network, names_output):
        network_nx = ig_to_nx(network, directed=True)
        components = [component for component in nx.weakly_connected_components(network_nx)]
        #comp_dist = [len(x) for x in components]
        gcc = []
        for component in components: # for each component 
                if len(gcc) < len(component): # looking for max
                        gcc = list(component)
        ids_to_delete = components
        ids_to_delete.remove(set(gcc)) # remove giant connected component to get ids to delete
        ids_to_delete = [y for x in ids_to_delete for y in x] # flatten list of sets to a list
        if names_output:
            ids_to_delete_names = []
            for v in network.vs:
                if v.index in ids_to_delete:
                    ids_to_delete_names.append(v['name'])
            return ids_to_delete_names
        return ids_to_delete

# Function for getting gcc from the network
def get_gcc(network_dict):
        network, network_oslom = network_dict['igraph'], network_dict['tuple']
        ids_to_delete = get_ids_to_delete(network, names_output=False)
        ids_to_delete_names = get_ids_to_delete(network, names_output=True)
        # delete in igraph
        network_gcc = network.copy() # copy network to have a backup
        network_gcc.delete_vertices(ids_to_delete) # delete ids that are not in gcc
        # delete in tuple
        network_gcc_oslom = [t for t in network_oslom if str(t[0]) not in ids_to_delete_names and str(t[1]) not in ids_to_delete_names]
        return {'igraph': network_gcc, 'tuple': network_gcc_oslom}


######################################################## MODULARITY CALCULATIONS (for infomap and oslom) ########################################################

# Function for getting adj. matrix
def get_sparse_adjacency_matrix(G, attr=None):
  if attr:
    source, target, data = zip(*[(e.source, e.target, e[attr]) 
      for e in G.es if not np.isnan(e[attr])])
  else:
    source, target = zip(*[(e.source, e.target) 
      for e in G.es])
    data = np.ones(len(source)).astype('int').tolist()
  if not G.is_directed():
    # If not directed, also create the other edge
    source, target = source + target, target + source;
    data = data + data
  L = coo_matrix((data, (source, target)), shape=[G.vcount(), G.vcount()]);
  return L


# Function for calculating modularity
def get_modularity(network_ig, community_dict):
    Q = 0
    G = network_ig.copy()
    A = get_sparse_adjacency_matrix(G, attr = 'weight')
    A = A.tocsr()
    if not G.is_directed():
        # for undirected graphs, in and out treated as the same thing
        out_strength = in_strength = {v.index:G.strength(v, weights='weight') for v in G.vs}
        M = sum(in_strength.values())
    elif G.is_directed():
        in_strength = {v.index:G.strength(v,mode="IN", weights ='weight') for v in G.vs}
        out_strength = {v.index:G.strength(v,mode="OUT", weights ='weight') for v in G.vs}
        M = sum(in_strength.values())
    else:
        print('Invalid graph type')
        raise TypeError
    nodes = G.vs.indices
    Q = np.sum([A[i,j] - in_strength[nodes[i]]*\
                         out_strength[nodes[j]]/M\
                 for i, j in product(range(len(nodes)),\
                                     range(len(nodes))) \
                if community_dict[nodes[i]] == community_dict[nodes[j]]])
    return Q / M

