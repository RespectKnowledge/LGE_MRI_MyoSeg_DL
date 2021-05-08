# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:34:57 2021

@author: Abdul Qayyum
"""

#%%
import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\statcomarpison'
dicep1=np.load(os.path.join(path,'Dicep1.npy'))
avgDice = float(sum(dicep1))/len(dicep1)
print(avgDice)
dicep2=np.load(os.path.join(path,'Dicep2.npy'))
avgDice2 = float(sum(dicep2))/len(dicep2)
print(avgDice2)
dicep3=np.load(os.path.join(path,'Dicep3.npy'))
avgDice3 = float(sum(dicep3))/len(dicep3)
print(avgDice3)
dicep4=np.load(os.path.join(path,'Dicep4.npy'))
avgDice4 = float(sum(dicep4))/len(dicep4)
print(avgDice4)
dicep7=np.load(os.path.join(path,'Dicep7.npy'))
avgDice7 = float(sum(dicep7))/len(dicep7)
print(avgDice7)
dicep8=np.load(os.path.join(path,'Dicep8.npy'))
avgDice8 = float(sum(dicep8))/len(dicep8)
print(avgDice8)
import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

list=[dicep1,dicep4,dicep7,dicep2,dicep3,dicep8]

df = pd.DataFrame(list)
df = df.transpose()
df.columns = ['Dicem1','dicem2','dicem3','dicem4','dicem5','dicem6']
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
#g= sns.violinplot(data=df,palette="Set1",linewidth=2)
#g.set_xticks(range(len(df))) # <--- set the ticks first
g.set_xticklabels(['DiceM1','DiceM2','DiceM3','DiceM4','DiceM5','DiceM6'],fontsize = 10)
g.set_ylabel("Dice(%)",fontsize=16)
g.set_xlabel("DL models comparisons for test cases",fontsize=16)



# fig_dims = (10, 6)
# fig, ax = plt.subplots(figsize=fig_dims)
# # g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# # g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# #g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
# g= sns.violinplot(data=df,palette="Set1",linewidth=2)
# g.set_xticks(range(len(df))) # <--- set the ticks first
# g.set_xticklabels(['accuracy','precision','recall','f1score'],fontsize = 13)



########### HD ##############################################

import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\statcomarpison'
HDp1=np.load(os.path.join(path,'HDp1.npy'))
avgHD = float(sum(HDp1))/len(HDp1)
print(avgHD)
HDp2=np.load(os.path.join(path,'HDp2.npy'))
avgHD2 = float(sum(HDp2))/len(HDp2)
print(avgHD2)
HDp3=np.load(os.path.join(path,'HDp3.npy'))
avgHD3 = float(sum(HDp3))/len(HDp3)
print(avgHD3)
HDp4=np.load(os.path.join(path,'HDp4.npy'))
avgHD4 = float(sum(HDp4))/len(HDp4)
print(avgHD4)
HDp7=np.load(os.path.join(path,'HDp7.npy'))
avgHD7 = float(sum(HDp7))/len(HDp7)
print(avgHD7)
HDp8=np.load(os.path.join(path,'HDp8.npy'))
avgHD8 = float(sum(HDp8))/len(HDp8)
print(avgHD8)

import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

list=[HDp4,HDp1,HDp3,HDp2,HDp7,HDp8]

df = pd.DataFrame(list)
df = df.transpose()
df.columns = ['Dicem1','dicem2','dicem3','dicem4','dicem5','dicem6']
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
#g= sns.violinplot(data=df,palette="Set1",linewidth=2)
#g.set_xticks(range(len(df))) # <--- set the ticks first
g.set_xticklabels(['HDM1','HDM2','HDM3','HDM4','HDM5','HDM6'],fontsize = 10)
g.set_ylabel("HD(mm)",fontsize=16)
g.set_xlabel("DL models comparisons for test cases",fontsize=16)

#############################################P and GT ##################################


import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\statcomarpison'
GTp1=np.load(os.path.join(path,'GTvolumep1.npy'))
GTp2=np.load(os.path.join(path,'GTvolumep2.npy'))
GTp3=np.load(os.path.join(path,'GTvolumep3.npy'))
GTp4=np.load(os.path.join(path,'GTvolumep4.npy'))
GTp7=np.load(os.path.join(path,'GTvolumep7.npy'))
GTp8=np.load(os.path.join(path,'GTvolumep8.npy'))

Pp1=np.load(os.path.join(path,'Predvolumep1.npy'))
Pp2=np.load(os.path.join(path,'Predvolumep2.npy'))
Pp3=np.load(os.path.join(path,'Predvolumep3.npy'))
Pp4=np.load(os.path.join(path,'Predvolumep4.npy'))
Pp7=np.load(os.path.join(path,'Predvolumep7.npy'))
Pp8=np.load(os.path.join(path,'Predvolumep8.npy'))

from scipy.stats import ttest_ind
import numpy as np
# data1=GTp2
# data2=Pp2
####################
ttest1,pval1 = ttest_ind(GTp1,Pp1)
print("p-value",pval1)
if pval1 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval2 = ttest_ind(GTp2,Pp2)
print("p-value",pval2)
if pval2 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval3 = ttest_ind(GTp3,Pp3)
print("p-value",pval3)
if pval3 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval4 = ttest_ind(GTp4,Pp4)
print("p-value",pval4)
if pval4 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval7 = ttest_ind(GTp7,Pp7)
print("p-value",pval7)
if pval7 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
  
ttest,pval8 = ttest_ind(GTp8,Pp8)
print("p-value",pval8)
if pval8 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")

#%%
import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\stateresults'
dicep1=np.load(os.path.join(path,'Dicep1.npy'))
avgDice = float(sum(dicep1))/len(dicep1)
print(avgDice)
dicep2=np.load(os.path.join(path,'Dicep2.npy'))
avgDice2 = float(sum(dicep2))/len(dicep2)
print(avgDice2)
dicep3=np.load(os.path.join(path,'Dicep3.npy'))
avgDice3 = float(sum(dicep3))/len(dicep3)
print(avgDice3)
dicep4=np.load(os.path.join(path,'Dicep4.npy'))
avgDice4 = float(sum(dicep4))/len(dicep4)
print(avgDice4)
dicep7=np.load(os.path.join(path,'Dicep5.npy'))
avgDice7 = float(sum(dicep7))/len(dicep7)
print(avgDice7)
dicep8=np.load(os.path.join(path,'Dicep6.npy'))
avgDice8 = float(sum(dicep8))/len(dicep8)
print(avgDice8)
import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dicep31=[0.806528,0.825566,0.819522,0.87091,0.7522,0.91028,0.92,0.95,0.93,0.97]
avgDice81 = float(sum(dicep31))/len(dicep31)
print(avgDice81)
dicep71=[0.805025,0.815731,0.849534,0.767675,0.744513,0.859328,0.933362,0.97,0.96,0.98]
avgDice71 = float(sum(dicep71))/len(dicep71)
print(avgDice71)

list=[dicep2,dicep31,dicep4,dicep1,dicep71,dicep8]

df = pd.DataFrame(list)
df = df.transpose()
df.columns = ['Dicem1','dicem2','dicem3','dicem4','dicem5','dicem6']
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
#g= sns.violinplot(data=df,palette="Set1",linewidth=2)
#g.set_xticks(range(len(df))) # <--- set the ticks first
g.set_xticklabels(['DiceM1','DiceM2','DiceM3','DiceM4','DiceM5','DiceM6'],fontsize = 10)
g.set_ylabel("Dice(%)",fontsize=16)
g.set_xlabel("DL models comparisons for test cases",fontsize=16)



# fig_dims = (10, 6)
# fig, ax = plt.subplots(figsize=fig_dims)
# # g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# # g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# #g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
# g= sns.violinplot(data=df,palette="Set1",linewidth=2)
# g.set_xticks(range(len(df))) # <--- set the ticks first
# g.set_xticklabels(['accuracy','precision','recall','f1score'],fontsize = 13)



########### HD ##############################################

import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\stateresults'
HDp1=np.load(os.path.join(path,'HDp1.npy'))
avgHD = float(sum(HDp1))/len(HDp1)
print(avgHD)
HDp2=np.load(os.path.join(path,'HDp2.npy'))
avgHD2 = float(sum(HDp2))/len(HDp2)
print(avgHD2)
HDp3=np.load(os.path.join(path,'HDp3.npy'))
avgHD3 = float(sum(HDp3))/len(HDp3)
print(avgHD3)
HDp4=np.load(os.path.join(path,'HDp4.npy'))
avgHD4 = float(sum(HDp4))/len(HDp4)
print(avgHD4)
HDp7=np.load(os.path.join(path,'HDp5.npy'))
avgHD7 = float(sum(HDp7))/len(HDp7)
print(avgHD7)
HDp8=np.load(os.path.join(path,'HDp6.npy'))
avgHD8 = float(sum(HDp8))/len(HDp8)
print(avgHD8)

import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

list=[HDp7,HDp4,HDp2,HDp8,HDp1,HDp3]

df = pd.DataFrame(list)
df = df.transpose()
df.columns = ['Dicem1','dicem2','dicem3','dicem4','dicem5','dicem6']
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
#g= sns.violinplot(data=df,palette="Set1",linewidth=2)
#g.set_xticks(range(len(df))) # <--- set the ticks first
g.set_xticklabels(['HDM1','HDM2','HDM3','HDM4','HDM5','HDM6'],fontsize = 10)
g.set_ylabel("HD(mm)",fontsize=16)
g.set_xlabel("DL models comparisons for test cases",fontsize=16)

#############################################P and GT ##################################


import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\stateresults'
GTp1=np.load(os.path.join(path,'GTvolumep1.npy'))
GTp2=np.load(os.path.join(path,'GTvolumep2.npy'))
GTp3=np.load(os.path.join(path,'GTvolumep3.npy'))
GTp4=np.load(os.path.join(path,'GTvolumep4.npy'))
GTp7=np.load(os.path.join(path,'GTvolumep5.npy'))
GTp8=np.load(os.path.join(path,'GTvolumep6.npy'))

Pp1=np.load(os.path.join(path,'Predvolumep1.npy'))
Pp2=np.load(os.path.join(path,'Predvolumep2.npy'))
Pp3=np.load(os.path.join(path,'Predvolumep3.npy'))
Pp4=np.load(os.path.join(path,'Predvolumep4.npy'))
Pp7=np.load(os.path.join(path,'Predvolumep5.npy'))
Pp8=np.load(os.path.join(path,'Predvolumep6.npy'))


from scipy.stats import ttest_ind
import numpy as np
####################
ttest1,pval1 = ttest_ind(GTp1,Pp1)
print("p-value",pval1)
if pval1 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval2 = ttest_ind(GTp2,Pp2)
print("p-value",pval2)
if pval2 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval3 = ttest_ind(GTp3,Pp3)
print("p-value",pval3)
if pval3 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval4 = ttest_ind(GTp4,Pp4)
print("p-value",pval4)
if pval4 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval7 = ttest_ind(GTp7,Pp7)
print("p-value",pval7)
if pval7 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
  
ttest,pval8 = ttest_ind(GTp8,Pp8)
print("p-value",pval8)
if pval8 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
#%%
import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\statmodels'
dicep1=np.load(os.path.join(path,'Dicep1.npy'))
avgDice = float(sum(dicep1))/len(dicep1)
print(avgDice)
dicep2=np.load(os.path.join(path,'Dicep2.npy'))
avgDice2 = float(sum(dicep2))/len(dicep2)
print(avgDice2)
dicep3=np.load(os.path.join(path,'Dicep3.npy'))
avgDice3 = float(sum(dicep3))/len(dicep3)
print(avgDice3)
dicep4=np.load(os.path.join(path,'Dicep4.npy'))
avgDice4 = float(sum(dicep4))/len(dicep4)
print(avgDice4)
dicep5=np.load(os.path.join(path,'Dicep5.npy'))
avgDice5 = float(sum(dicep5))/len(dicep5)
print(avgDice5)
dicep6=np.load(os.path.join(path,'Dicep6.npy'))
avgDice6 = float(sum(dicep6))/len(dicep6)
print(avgDice6)

dicep71=np.load(os.path.join(path,'Dicep71.npy'))
avgDice71 = float(sum(dicep71))/len(dicep71)
print(avgDice71)

dicep8=np.load(os.path.join(path,'Dicep8.npy'))
avgDice8 = float(sum(dicep8))/len(dicep8)
print(avgDice8)


import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# dicep31=[0.806528,0.825566,0.819522,0.87091,0.7522,0.91028,0.92,0.95,0.93,0.97]
# avgDice81 = float(sum(dicep31))/len(dicep31)
# print(avgDice81)
# dicep71=[0.805025,0.815731,0.849534,0.767675,0.744513,0.859328,0.933362,0.97,0.96,0.98]
# avgDice71 = float(sum(dicep71))/len(dicep71)
# print(avgDice71)

list=[dicep5,dicep1,dicep4,dicep6,dicep2,dicep8,dicep71,dicep3]

df = pd.DataFrame(list)
df = df.transpose()
df.columns = ['Dicem1','dicem2','dicem3','dicem4','dicem5','dicem6','dicem7','dicem8']
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
#g= sns.violinplot(data=df,palette="Set1",linewidth=2)
#g.set_xticks(range(len(df))) # <--- set the ticks first
g.set_xticklabels(['DiceM1','DiceM2','DiceM3','DiceIntra','DiceInter','DiceM4','DiceM5','DiceM6'],fontsize = 10)
g.set_ylabel("Dice(%)",fontsize=16)
g.set_xlabel("DL models comparisons for test cases",fontsize=16)



# fig_dims = (10, 6)
# fig, ax = plt.subplots(figsize=fig_dims)
# # g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# # g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# #g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
# g= sns.violinplot(data=df,palette="Set1",linewidth=2)
# g.set_xticks(range(len(df))) # <--- set the ticks first
# g.set_xticklabels(['accuracy','precision','recall','f1score'],fontsize = 13)



########### HD ##############################################

import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\statmodels'
HDp1=np.load(os.path.join(path,'HDp1.npy'))
avgHD = float(sum(HDp1))/len(HDp1)
print(avgHD)
HDp2=np.load(os.path.join(path,'HDp2.npy'))
avgHD2 = float(sum(HDp2))/len(HDp2)
print(avgHD2)
HDp3=np.load(os.path.join(path,'HDp3.npy'))
avgHD3 = float(sum(HDp3))/len(HDp3)
print(avgHD3)
HDp4=np.load(os.path.join(path,'HDp4.npy'))
avgHD4 = float(sum(HDp4))/len(HDp4)
print(avgHD4)
HDp5=np.load(os.path.join(path,'HDp5.npy'))
avgHD5 = float(sum(HDp5))/len(HDp5)
print(avgHD5)
HDp6=np.load(os.path.join(path,'HDp6.npy'))
avgHD6 = float(sum(HDp6))/len(HDp6)
print(avgHD6)
HDp7=np.load(os.path.join(path,'HDp7.npy'))
avgHD7 = float(sum(HDp7))/len(HDp7)
print(avgHD7)

HDp8=np.load(os.path.join(path,'HDp8.npy'))
avgHD8 = float(sum(HDp8))/len(HDp8)
print(avgHD8)

import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

list=[HDp5,HDp8,HDp7,HDp4,HDp1,HDp6,HDp3,HDp2]

df = pd.DataFrame(list)
df = df.transpose()
df.columns = ['Dicem1','dicem2','dicem3','dicem4','dicem5','dicem6','dicem7','dicem8']
fig_dims = (10, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# g1=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
# g=sns.violinplot(x="variable", y="Value", data=pd.melt(df),palette="Set3",linewidth=2.5,color='red')
g= sns.boxplot(data=df,palette="Set1",linewidth=2.5)
#g= sns.violinplot(data=df,palette="Set1",linewidth=2)
#g.set_xticks(range(len(df))) # <--- set the ticks first
g.set_xticklabels(['HDM1','HDM2','HDM3','HDIntra','HDInter','HDM4','HDM5','HDM6'],fontsize = 10)
g.set_ylabel("HD(mm)",fontsize=16)
g.set_xlabel("DL models comparisons for test cases",fontsize=16)

#############################################P and GT ##################################


import numpy as np
import os
path='D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\statmodels'
GTp1=np.load(os.path.join(path,'GTvolumep1.npy'))
GTp2=np.load(os.path.join(path,'GTvolumep2.npy'))
GTp3=np.load(os.path.join(path,'GTvolumep3.npy'))
GTp4=np.load(os.path.join(path,'GTvolumep4.npy'))
GTp5=np.load(os.path.join(path,'GTvolumep5.npy'))
GTp6=np.load(os.path.join(path,'GTvolumep6.npy'))
GTp7=np.load(os.path.join(path,'GTvolumep7.npy'))
GTp8=np.load(os.path.join(path,'GTvolumep8.npy'))

Pp1=np.load(os.path.join(path,'Predvolumep1.npy'))
Pp2=np.load(os.path.join(path,'Predvolumep2.npy'))
Pp3=np.load(os.path.join(path,'Predvolumep3.npy'))
Pp4=np.load(os.path.join(path,'Predvolumep4.npy'))
Pp5=np.load(os.path.join(path,'Predvolumep5.npy'))
Pp6=np.load(os.path.join(path,'Predvolumep6.npy'))
Pp7=np.load(os.path.join(path,'Predvolumep7.npy'))
Pp8=np.load(os.path.join(path,'Predvolumep8.npy'))


from scipy.stats import ttest_ind
import numpy as np
####################
ttest1,pval1 = ttest_ind(GTp1,Pp1)
print("p-value",pval1)
if pval1 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval2 = ttest_ind(GTp2,Pp2)
print("p-value",pval2)
if pval2 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval3 = ttest_ind(GTp3,Pp3)
print("p-value",pval3)
if pval3 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval4 = ttest_ind(GTp4,Pp4)
print("p-value",pval4)
if pval4 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")

ttest,pval5 = ttest_ind(GTp5,Pp5)
print("p-value",pval5)
if pval5 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval6 = ttest_ind(GTp6,Pp6)
print("p-value",pval6)
if pval6 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
ttest,pval7 = ttest_ind(GTp7,Pp7)
print("p-value",pval7)
if pval7 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
  
ttest,pval8 = ttest_ind(GTp8,Pp8)
print("p-value",pval8)
if pval8 <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  


#%% bandaltman plot between surface area between GT and predicted segmentation map
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Set default Seaborn preferences (disabled Pingouin >= 0.3.4)
# See https://github.com/raphaelvallat/pingouin/issues/85
# sns.set(style='ticks', context='notebook')

__all__ = ["plot_blandaltman", "qqplot", "plot_paired",
           "plot_shift", "plot_rm_corr", "plot_circmean"]


def plot_blandaltman(x, y, agreement=1.96, confidence=.95, figsize=(5, 4),
                     dpi=100, ax=None):
    """
    Generate a Bland-Altman plot to compare two sets of measurements.
    Parameters
    ----------
    x, y : np.array or list
        First and second measurements.
    agreement : float
        Multiple of the standard deviation to plot limit of agreement bounds.
        The defaults is 1.96.
    confidence : float
        If not ``None``, plot the specified percentage confidence interval on
        the mean and limits of agreement.
    figsize : tuple
        Figsize in inches
    dpi : int
        Resolution of the figure in dots per inches.
    ax : matplotlib axes
        Axis on which to draw the plot
    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.
    Notes
    -----
    Bland-Altman plots [1]_ are extensively used to evaluate the agreement
    among two different instruments or two measurements techniques.
    They allow identification of any systematic difference between the
    measurements (i.e., fixed bias) or possible outliers.
    The mean difference is the estimated bias, and the SD of the differences
    measures the random fluctuations around this mean. If the mean value of the
    difference differs significantly from 0 on the basis of a 1-sample t-test,
    this indicates the presence of fixed bias. If there is a consistent bias,
    it can be adjusted for by subtracting the mean difference from the new
    method.
    It is common to compute 95% limits of agreement for each comparison
    (average difference ± 1.96 standard deviation of the difference), which
    tells us how far apart measurements by 2 methods were more likely to be
    for most individuals. If the differences within mean ± 1.96 SD are not
    clinically important, the two methods may be used interchangeably.
    The 95% limits of agreement can be unreliable estimates of the population
    parameters especially for small sample sizes so, when comparing methods
    or assessing repeatability, it is important to calculate confidence
    intervals for 95% limits of agreement.
    The code is an adaptation of the
    `PyCompare <https://github.com/jaketmp/pyCompare>`_ package. The present
    implementation is a simplified version; please refer to the original
    package for more advanced functionalities.
    References
    ----------
    .. [1] Bland, J. M., & Altman, D. (1986). Statistical methods for assessing
           agreement between two methods of clinical measurement. The lancet,
           327(8476), 307-310.
    Examples
    --------
    Bland-Altman plot
    .. plot::
        >>> import numpy as np
        >>> import pingouin as pg
        >>> np.random.seed(123)
        >>> mean, cov = [10, 11], [[1, 0.8], [0.8, 1]]
        >>> x, y = np.random.multivariate_normal(mean, cov, 30).T
        >>> ax = pg.plot_blandaltman(x, y)
    """
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    n = x.size
    mean = np.vstack((x, y)).mean(0)
    diff = x - y
    md = diff.mean()
    sd = diff.std(axis=0, ddof=1)

    # Confidence intervals
    if confidence is not None:
        assert 0 < confidence < 1
        ci = dict()
        ci['mean'] = stats.norm.interval(confidence, loc=md,
                                         scale=sd / np.sqrt(n))
        seLoA = ((1 / n) + (agreement**2 / (2 * (n - 1)))) * (sd**2)
        loARange = np.sqrt(seLoA) * stats.t.ppf((1 - confidence) / 2, n - 1)
        ci['upperLoA'] = ((md + agreement * sd) + loARange,
                          (md + agreement * sd) - loARange)
        ci['lowerLoA'] = ((md - agreement * sd) + loARange,
                          (md - agreement * sd) - loARange)

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot the mean diff, limits of agreement and scatter
    ax.axhline(md, color='#6495ED', linestyle='--')
    ax.axhline(md + agreement * sd, color='coral', linestyle='--')
    ax.axhline(md - agreement * sd, color='coral', linestyle='--')
    ax.scatter(mean, diff, alpha=0.5)

    loa_range = (md + (agreement * sd)) - (md - agreement * sd)
    offset = (loa_range / 100.0) * 1.5

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(0.98, md + offset, 'Mean', ha="right", va="bottom",
            transform=trans)
    ax.text(0.98, md - offset, '%.2f' % md, ha="right", va="top",
            transform=trans)

    ax.text(0.98, md + (agreement * sd) + offset, '+%.2f SD' % agreement,
            ha="right", va="bottom", transform=trans)
    ax.text(0.98, md + (agreement * sd) - offset,
            '%.2f' % (md + agreement * sd), ha="right", va="top",
            transform=trans)

    ax.text(0.98, md - (agreement * sd) - offset, '-%.2f SD' % agreement,
            ha="right", va="top", transform=trans)
    ax.text(0.98, md - (agreement * sd) + offset,
            '%.2f' % (md - agreement * sd), ha="right", va="bottom",
            transform=trans)

    if confidence is not None:
        ax.axhspan(ci['mean'][0], ci['mean'][1],
                   facecolor='#6495ED', alpha=0.2)

        ax.axhspan(ci['upperLoA'][0], ci['upperLoA'][1],
                   facecolor='coral', alpha=0.2)

        ax.axhspan(ci['lowerLoA'][0], ci['lowerLoA'][1],
                   facecolor='coral', alpha=0.2)

    # Labels and title
    ax.set_ylabel('VA-VM (ml)')
    ax.set_xlabel('(VA-VM)/2 (ml)')
    ax.set_title('Bland-Altman plot')

    # Despine and trim
    sns.despine(trim=True, ax=ax)

    return ax
GTp1=np.load(os.path.join(path,'GTvolumep1.npy'))
GTp2=np.load(os.path.join(path,'GTvolumep2.npy'))
GTp3=np.load(os.path.join(path,'GTvolumep3.npy'))
GTp4=np.load(os.path.join(path,'GTvolumep4.npy'))
GTp7=np.load(os.path.join(path,'GTvolumep5.npy'))
GTp8=np.load(os.path.join(path,'GTvolumep6.npy'))

Pp1=np.load(os.path.join(path,'Predvolumep1.npy'))
Pp2=np.load(os.path.join(path,'Predvolumep2.npy'))
Pp3=np.load(os.path.join(path,'Predvolumep3.npy'))
Pp4=np.load(os.path.join(path,'Predvolumep4.npy'))
Pp7=np.load(os.path.join(path,'Predvolumep5.npy'))
Pp8=np.load(os.path.join(path,'Predvolumep6.npy'))
ax = plot_blandaltman(GTp1, Pp1)
ax = plot_blandaltman(GTp2, Pp2)
ax = plot_blandaltman(GTp3, Pp3)
ax = plot_blandaltman(GTp4, Pp4)
ax = plot_blandaltman(GTp5, Pp5)
ax = plot_blandaltman(GTp6, Pp6)
ax = plot_blandaltman(GTp7, Pp7)
ax = plot_blandaltman(GTp8, Pp8)


#%%
import numpy as np
import os
import shutil
import SimpleITK as sitk
import scipy.ndimage as ndimage
#pathGT="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\GT"
pathGT="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\Myochallenege\\Volumenii\\testmasknii"
#pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\maskbinaryvolume\\niftivolume\\M1"
pathPrediction="D:\\AQProject\\completedataset2020heart\\nifti\\nifti_main\\testcases\\allmodelpredictions\\HeartResults\\predictionmodel\\volumenii\\M1pred"


dice=[]
HD=[]
volumeDifference=[]
volumeDifferenceRate=[]
volumePrediction=[]
volumeGT=[]
for filePrediction in os.listdir(pathPrediction):
    print(filePrediction)
    #  load prediction mask as a nifiti, you can use nib.load as well for nifti
    prediction = sitk.ReadImage(os.path.join(pathPrediction, filePrediction), sitk.sitkInt16) 
    #  the prediction mask array should be one hot format
    predArray = sitk.GetArrayFromImage(prediction)  # convert into numpy array

    # load GT mask. 
    # You should modify the GT file name if its name is different to the prediction file
    GT = sitk.ReadImage(os.path.join(pathGT, filePrediction.replace('pred','GT')), sitk.sitkInt8) 
    GTArray = sitk.GetArrayFromImage(GT)
    spacing=GT.GetSpacing()
    #print(GTArray.shape)
    #print(predArray.shape)
    dice.append(dc(predArray, GTArray))
    aVolumePred=volume(predArray, spacing)
    aVolumeGT=volume(GTArray, spacing)
    volumePrediction.append(aVolumePred)
    volumeGT.append(aVolumeGT)
    volumeDifference.append(abs(aVolumePred-aVolumeGT))
    #print(volumeDifference)
    HD.append(hd(predArray, GTArray))

avgDice = float(sum(dice))/len(dice)
print(avgDice)
avgVD= float(sum(volumeDifference))/len(volumeDifference)
print(avgVD)
avgHd= float(sum(HD))/len(HD)
print(avgHd)