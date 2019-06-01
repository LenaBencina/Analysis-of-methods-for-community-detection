#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################################################################
### This file is used for converting csv or dat file to ncol.
##################################################################################################################################################################


"""
Created on Thu Oct 25 15:29:37 2018

@author: Lena
"""
import os 
import pandas as pd

os.chdir("/Users/miha/Desktop/Mag./Podatki/")

def csv_to_ncol(file_name, sep):
    network_csv = pd.read_csv(file_name + '.csv', sep=sep) 
    network_csv.to_csv(file_name + '.txt', sep=' ', index=False, header=False)
    print('file saved')
    

def dat_to_ncol(file_name, sep):
    network_csv = pd.read_csv(file_name + '.dat', sep=sep) 
    network_csv.to_csv(file_name + '.txt', sep=' ', index=False, header=False)
    print('file saved')


#csv_to_ncol('EP/edges_from[oct2014]_to[sep2015]_lang[all]_meps[only]', sep=',')
#csv_to_ncol('EP/edges_from[oct2014]_to[sep2015]_lang[all]_meps[ext]', sep=',')
#csv_to_ncol('EMAIL/email-edges', sep=' ')
