#!/usr/bin/env python 
import numpy as np
from matplotlib import pyplot as plt
from glob import glob 
import os 
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
import pandas as pd
import random
from scipy.stats import pearsonr
import h5py
import matplotlib
import matplotlib.ticker as ticker
import sys 

norm = mcolors.TwoSlopeNorm(vmin=-1, vmax =60, vcenter=0)

def spine_set(ax0,val):
    ax0.spines['bottom'].set_linewidth(val)
    ax0.spines['left'].set_linewidth(val)
    ax0.spines['right'].set_linewidth(val)
    ax0.spines['top'].set_linewidth(val)

def tick_set(ax0):
    ax0.tick_params(axis='both', direction='out', width=2)
    return

def tick_range_set(ax0, x_val_min, x_val_max, x_val_ran, y_val_min, y_val_max, y_val_ran):
    ax0.set_xlim(x_val_min, x_val_max)
    ax0.set_ylim(y_val_min, y_val_max)
    ax0.set_xticks(x_val_ran)
    ax0.set_yticks(y_val_ran)
    ax0.tick_params(axis='both', labelsize=12)
    return

def title_set(ax0, x, y, name0, name1, name2):
    #ax0.legend()
    #ax0.text(x, y, name0, fontsize=12)
    ax0.set_xlabel(name1, fontsize=12)
    ax0.set_ylabel(name2, fontsize=12)
    return



def gencom(coord, symbol, f_name):
    with open(f_name, 'w') as f:
        f.write('# pm6 \n\n')
        f.write('ddd \n\n')
        f.write('0 1 \n')
        for ss, cc in zip(symbol, coord):
            f.write('%s %f %f %f \n' % (ss, cc[0], cc[1], cc[2]))
        f.write('\n')

f_name = sys.argv[1]
data = np.load(f_name, allow_pickle=True)
png_name = os.path.basename(f_name).split('.')[0]
dis = data['dis']; E2_label = data['E2_label']; E2_nn = data['E2_nn']
coord_A = data['coord_A']; coord_B = data['coord_B']
symbol_A = data['symol_A']; symbol_B = data['symol_B']
E1_pol = data['E1pol_label']; E1pol_nn = data['E1pol_nn']
E2_disp = data['E2disp_label']; E2disp_nn = data['E2disp_nn']
E_tot = E1_pol + E2_label + E2_disp
E_tot_nn = E1pol_nn + E2_nn + E2disp_nn

dis_ord = np.argsort(dis)
dis = dis[dis_ord]

E2_label = E2_label[dis_ord]; E2_nn = E2_nn[dis_ord]
E1_pol = E1_pol[dis_ord]; E1pol_nn = E1pol_nn[dis_ord]
E2_disp = E2_disp[dis_ord]; E2disp_nn = E2disp_nn[dis_ord]
E_tot = E_tot[dis_ord]; E_tot_nn = E_tot_nn[dis_ord]

#matplotlib.rcParams['font.family'] = 'Arial'
fig,axes = plt.subplots(nrows=1, ncols=1, figsize=(3.4,3.0))
ax1 = axes

ax1.plot(dis, E2_label, color='#FCBB44', linestyle='-', label='E_tot', linewidth=2)
ax1.plot(dis, E2_nn, color ='#FCBB44', linestyle='--', linewidth=2)
ax1.scatter(dis, E2_label, c='#FCBB44', s=20)
ax1.scatter(dis, E2_nn, c='#FCBB44', s=20)

ax1.set_xlabel('Distance (Angstrom)', fontsize=14)
ax1.tick_params(axis='both', labelsize=14, direction='out', width=2)
#ax1.xaxis.set_major_locator(ticker.MaxNLocator(4))
#ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
for spine in ax1.spines.values():
    spine.set_linewidth(2) 
#ax1.legend(loc='upper right', fontsize=14, frameon=False)

ax2 = ax1.twinx()
ax2.plot(dis, E2_disp, color = '#F1766D', linestyle='-', label='E_es', linewidth=2)
ax2.plot(dis, E2disp_nn, color='#F1766D', linestyle='--', linewidth=2)
ax2.scatter(dis, E2_disp, c='#F1766D', s=20)
ax2.scatter(dis, E2disp_nn, c='#F1766D', s=20)
#ax2.legend(loc='upper right', fontsize=14, frameon=False)

ax2.set_ylabel('E (kcal/mol)', fontsize=14)#, color=(150/255,187/255,213/255))
ax1.set_ylabel('E (kcal/mol)', fontsize=14)#(, color=(180/255,227/255,211/255))

ax2.tick_params(axis='both', labelsize=14, direction='out', width=2)

#ax2.xaxis.set_major_locator(ticker.MaxNLocator(4))
#ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
ax1.set_xlim([1.5,6.0])
ax2.set_xlim([1.5,6.0])
#ax1.x_val_min = 1.5
#ax1.x_val_max = 6.0
ax1.set_xticks([1.5, 3.0, 4.5, 6.0])  
#ax1.set_yticks([6,12,18,24]) 
#ax2.x_val_max = 6.0
#ax2.x_val_min = 1.5
ax2.set_xticks([1.5, 3.0, 4.5, 6.0]) 
# 59_59
#ax1.set_ylim([-9,0])
#ax2.set_ylim([-0.3,0.0])
#ax1.set_yticks([-9,-6,-3,0])  
#ax2.set_yticks([-0.3,-0.2,-0.1,0.0]) 
#ax1.set_yticks([8,16,24,32]) 

# 123_123
#ax1.set_ylim([-21,1])
#ax2.set_ylim([-1.5,0])
#ax1.set_yticks([-21,-14, -7, 0])  
#ax2.set_yticks([-1.5,-1.0,-0.5,0])  
#ax2.set_yticks([21,25,29,33]) 
#ax1.set_yticks([21,25,29,33]) 

# 142_142
#ax1.set_ylim([-33,0])
#ax2.set_ylim([-2,1])
#ax1.set_yticks([-33,-22,-11,0])  
#ax2.set_yticks([-2,-1,0,1])  
#ax2.set_yticks([-50,-35,-20,-5]) 

# 180_180
#ax1.set_ylim([-24,0])
#ax2.set_ylim([-1.5,0])
#ax1.set_yticks([-24,-16,-8,0])  
#ax2.set_yticks([-1.5,-1.0,-0.5,0])  
#ax2.set_yticks([-7,-5,-3,-1]) 

# 104_104
ax1.set_ylim([-9,0])
ax2.set_ylim([-0.5,0.1])
ax1.set_yticks([-9,-6,-3,0])  
ax2.set_yticks([-0.5,-0.3,-0.1,0.1])  
#ax2.set_yticks([-13,-9,-5,-1])  
#ax2.set_yticks([-7,-5,-3,-1]) 


for spine in ax2.spines.values():
    spine.set_linewidth(2) 

plt.xlabel('distance', fontsize=14)
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.20, right=0.80)
plt.savefig(f'{png_name}_tot.png',dpi=600)
plt.close()    

