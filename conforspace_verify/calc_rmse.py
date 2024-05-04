#!/usr/bin/env python 
import os
from glob import glob 

cwd = os.getcwd()
f_dirs = glob('./conf.*/')
#f_dirs = ['conf.0']

def calc_rmse():
    for ii in range(500):
        os.system(f'calculate_rmsd frame_0.xyz frame_{ii}.xyz >> rmsd.dat')

for dir0 in f_dirs:
    print(dir0)
    os.chdir(dir0)
    calc_rmse()
    os.chdir(cwd)