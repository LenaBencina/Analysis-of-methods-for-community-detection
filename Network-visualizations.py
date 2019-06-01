#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################################################################
### This file contains everything related with network visualizations (Retweet network, Email network and all seven configurations of LFR network).
### Visualizations are made with PY3PLEX.
##################################################################################################################################################################

"""
Created on Wed Apr 10 11:12:32 2019

@author: Lena
"""

# Libraries
from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import all_color_names,colors_default
from py3plex.core import multinet

import matplotlib.pyplot as mplot
mplot.switch_backend('TkAgg')

################################################################################################################
# LFR

lfr1_network = multinet.multi_layer_network().load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network1.csv', directed=True, input_type='edgelist')
lfr7_network = multinet.multi_layer_network().load_network('/Users/miha/Desktop/Mag./Podatki/LFR/network7.csv', directed=True, input_type='edgelist')

#lfr1_network.basic_stats() ## check core imports
#lfr1_network.visualize_network() ## visualize
#plt.show()

network_colors, graph = lfr1_network.get_layers(style='hairball')
hairball_plot(graph, network_colors, legend=False)
plt.savefig('/Users/miha/Google Drive/Koda/Python/Visualizations/lfr1_network.png', dpi=500, bbox_inches = 'tight')
plt.clf()


network_colors, graph = lfr7_network.get_layers(style='hairball')
hairball_plot(graph, network_colors, legend=False)
plt.savefig('/Users/miha/Google Drive/Koda/Python/Visualizations/lfr7_network.png', dpi=500, bbox_inches = 'tight')
plt.clf()


################################################################################################################
# EU

# correct file format
eu_df = pd.read_csv('/Users/miha/Desktop/Mag./Podatki/EP/edges_from[oct2014]_to[sep2015]_lang[all]_meps[only].csv')
eu_df.to_csv('/Users/miha/Desktop/Mag./Podatki/EP/edges-visualization.csv', sep='\t', header=False, index=False)

eu_network = multinet.multi_layer_network().load_network('/Users/miha/Desktop/Mag./Podatki/EP/edges-visualization.csv', directed=True, input_type='edgelist')
network_colors, graph = eu_network.get_layers(style='hairball')
hairball_plot(graph, network_colors, legend=False, layout_parameters={'iterations':20})
plt.savefig('/Users/miha/Google Drive/Koda/Python/Visualizations/eu_network.png', dpi=500, bbox_inches = 'tight')
plt.clf()


################################################################################################################
# EMAIL

# hairball email network
email_network = multinet.multi_layer_network().load_network('/Users/miha/Desktop/Mag./Podatki/EMAIL/email-edges.csv', directed=True, input_type='edgelist')
network_colors, graph = email_network.get_layers(style='hairball')
hairball_plot(graph, network_colors, legend=False)
plt.savefig('/Users/miha/Google Drive/Koda/Python/Visualizations/email_network.png', dpi=500, bbox_inches = 'tight')
plt.clf()

# get gcc only
from CD-functions import *
network_email = load_network('/Users/miha/Desktop/Mag./Podatki/EMAIL/email-edges', via_ncol = True, weighted = True, sep = ' ')
network_email_gcc = get_gcc(network_email) # [986, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
df_gcc = pd.DataFrame(network_email_gcc['tuple'])
df_gcc.to_csv('/Users/miha/Desktop/Mag./Podatki/EMAIL/gcc-edges.csv', sep='\t', header=False, index=False)

# hairball email gcc network
gcc_network = multinet.multi_layer_network().load_network('/Users/miha/Desktop/Mag./Podatki/EMAIL/gcc-edges.csv', directed=True, input_type='edgelist')
network_colors, graph = gcc_network.get_layers(style='hairball')
hairball_plot(graph, network_colors, legend=False)
plt.savefig('/Users/miha/Google Drive/Koda/Python/Visualizations/email_network_gcc.png', dpi=500, bbox_inches = 'tight')
plt.clf()

# checking the differences between the original and the reduced one
email_network.basic_stats()
gcc_network.basic_stats()