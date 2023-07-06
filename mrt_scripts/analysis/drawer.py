#%%
import matplotlib.pyplot as plt 

total_data = "	0.937481239	0.765432915	0.913263235	0.872176012	0.878699249	0.880149996 0.937481239		0.752165091	0.954799225	0.942903998	0.934351152	0.934058427 0.765432915	0.752165091		0.729723447	0.670129653	0.673383126	0.67310447 0.913263235	0.954799225	0.729723447		0.953541872	0.950913508	0.949807107 0.872176012	0.942903998	0.670129653	0.953541872		0.980868605	0.982152925 0.878699249	0.934351152	0.673383126	0.950913508	0.980868605		0.997944441 0.880149996	0.934058427	0.67310447	0.949807107	0.982152925	0.997944441	".replace('\t', ' ')
total_data = total_data.split(' ')
total_data = [item for item in total_data if item]
data_in_matrix = []
idx = 0

for i in range(7):
    for j in range(7):
        if j == 0:
            data_in_matrix.append([])
        if i == j:
            data_in_matrix[-1].append(1)
        else:
            data_in_matrix[-1].append(float(total_data[idx]))
            idx += 1

print(data_in_matrix)

import pandas as pd #这里导入pandas计算包
import numpy as np #导入numpy计算包，没装包的需要先安装下
import seaborn as sns #导入画图包
import matplotlib.pyplot as plt 

corr_mat = data_in_matrix

f, ax = plt.subplots(figsize=(12, 8)) #定义画布的大小
x_data = ["BLEU","BERTScore","BARTScore","BLEURT","COMET","UniTE_ref","UniTE_src_ref"]

mask = np.zeros_like(corr_mat)

for i in range(1,len(mask)):

  for j in range(0,i):

    mask[i][j] = True# 掩盖掉上面的三角形
"""
plt.figure(dpi=120)
sns.heatmap(data=df,cmap=sns.cubehelix_palette(as_cmap=True) # #渐变色盘：sns.cubehelix_palette()使用           )plt.title("使用seaborn cubehelix颜色盘：sns.diverging_palette(220, 20, sep=20, as_cmap=True)")
"""
sns.heatmap(corr_mat, annot=True, fmt=".4f",mask=mask,linewidths=.05,square=True,annot_kws={'size':8}, cmap=sns.cubehelix_palette(as_cmap=True) )#显示相关性数值

plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)#设置画布边缘尺寸，可以自己调整plt.save('相关性.png',dpi=300)#设置图片地址这里是相对地址，图片保存为矢量图分辨率300，
plt.yticks(np.arange(len(x_data)) + 0.5, position=(1.05,0), fontsize=9, labels=x_data) 
plt.xticks(np.arange(len(x_data)) + 0.5, position=(0,1.05), fontsize=9, labels=x_data) 
plt.tick_params(top=False,bottom=False,left=False,right=False)

plt.show()#显示绘图内容
plt.savefig('mrt_analysis_results/all_plot_temp/metric_corr.jpg')
# %%

import numpy as np
import matplotlib.pyplot as plt
#准备数据
data = [[-100.00,-107.48,-20.33,14.96,-435.00,-423.12,-408.72],
        [0.00,0.52,-0.28,1.61,7.53,8.12,7.40],
        [-16.90,-3.23,-0.80,3.22,13.13,14.19,13.11]]
x_label = ["BLEU","BERTScore","BARTScore","BLEURT","COMET","UniTE_ref","UniTE_src_ref"]

X = np.arange(7)
fig = plt.figure()
#添加子图区域
ax = fig.add_axes([0,0,1,1])
#绘制柱状图
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25, label ="A")
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25, label ="B")
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25, label ="C")
plt.legend()
plt.xticks(np.arange(len(x_label)) + 0.5, fontsize=9, labels=x_label) 
plt.axhline(y=0,xmin=0,xmax=1,linestyle="-", color='k', lw=0.5)

for i in range(len(data[0])):
    for j in range(len(data)):
        plt.text(i + 0.25 * j,\
            data[j][i] + 10 if data[j][i] >= 0 else data[j][i] - 10, 
            "%s" % data[j][i], 
            va='center', 
            verticalalignment='center', 
            horizontalalignment='center',
            fontsize=6)  #显示y轴数据

# python3 ../drawer.py

# %%
