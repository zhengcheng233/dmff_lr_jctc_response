#!/usr/bin/env python 
import numpy as np
import os 
from matplotlib import pyplot as plt

confs = ['conf.67', 'conf.68', 'conf.175', 'conf.255', 'conf.260']

rmsds = []
for conf in confs:
    rmsd = np.loadtxt(os.path.join(conf, 'rmsd.dat'))
    rmsds.append(rmsd)
times = np.arange(0,50,0.1)
legend_name = ['(a)', '(b)', '(c)', '(d)', '(e)']

plt.figure(figsize=(10,15))
# 绘制每个子图，并添加图例
for i, rmsd in enumerate(rmsds):
    ax = plt.subplot(5, 1, i+1)
    ax.plot(times, rmsd, color='black', linewidth=2.0)
    ax.legend(loc='upper right',bbox_to_anchor=(1, 1),fontsize=16, framealpha=0.)  # 添加图例

    #ax.set_title(legend_name[i],   # 设置图例标签为 (a), (b), (c), (d), (e)
    ax.spines['top'].set_linewidth(2.0)  # 设置图形边框线宽
    ax.spines['right'].set_linewidth(2.0)  # 设置图形边框线宽
    ax.spines['bottom'].set_linewidth(2.0)  # 设置图形边框线宽
    ax.spines['left'].set_linewidth(2.0)  # 设置图形边框线宽
    ax.tick_params(axis='both', direction='out', width=2.0)
    ax.set_xlabel('Time (ps)', fontsize=16)  # 设置x轴名称及字号
    ax.set_ylabel('RMSD (angstrom)', fontsize=16)  # 设置y轴名称及字号
    ax.set_xticks([0,10,20,30,40,50])  # 设置x轴坐标刻度
    ax.set_yticks([0, 1.0, 2.0, 3.0])  # 设置y轴坐标刻度
    ax.tick_params(axis='both', which='major', labelsize=16)  # 设置坐标刻度的字体大小

plt.tight_layout()  # 调整子图间距
plt.savefig('rmsd.png', dpi=600)  # 保存图形

