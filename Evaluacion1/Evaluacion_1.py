# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:05:07 2019

@author: olguin aguilar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("vid18_180219.dat")

data1 = pd.DataFrame()

data1['TIMESTAMP'] = data['TIMESTAMP']
data1['AirTC_Avg'] = data['AirTC_Avg']

'''
data1['NTIMESTAMP'] =  pd.to_datetime(data1['TIMESTAMP'])
data1['Año'] = data1['NTIMESTAMP'].dt.year
data1['Mes'] = data1['NTIMESTAMP'].dt.month
data1['Dia'] = data1['NTIMESTAMP'].dt.day
'''
#########################################################

data2 = pd.DataFrame(data1)

data2 = data2.drop('TIMESTAMP', 1)

data2['Fecha'] = data['TIMESTAMP'].str.extract('(..-..-..)',expand=True)
print(data2.head())

fecha = data['TIMESTAMP'].str.extract('(..-..-..)',expand=True)
tmax = data2.groupby(['Fecha'])[['AirTC_Avg']].max()
tmin = data2.groupby(['Fecha'])[['AirTC_Avg']].min()
fecha
tmax
df = pd.DataFrame()
df['Fecha'] = fecha
df.head()



df = df.assign(Tmax=tmax.values)
df = df.assign(Tmin=tmin.values)
df

'''
data2['Tmax'] = data1.groupby(['Dia'])[['AirTC_Avg']].max()
data2['Tmin'] = data1.groupby(['Dia'])[['AirTC_Avg']].min()
data2['Tprom'] = data1.groupby(['Dia'])[['AirTC_Avg']].mean()
'''
