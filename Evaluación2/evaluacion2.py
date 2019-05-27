# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:06:53 2019

@author: olguin aguilar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

dat = pd.DataFrame(pd.read_csv("meteo-vid-2018.csv", engine="python"))
dat.head()

#Creamos una tabla con los valores especificados
#filtramos las columnas del archivo de datos original
dat1=dat.filter(['Date','Time','AirTC_Avg','RH','WS_ms_S_WVT','Rn_Avg','albedo_Avg'],axis=1)
dat1=pd.DataFrame(dat1)
dat1.head()


#extraemos año, mes y dia de la columna Date y los agregamos al data frame como columnas

#Creamos la variable FECHA a partir de Date y Time. Desechamos dichas columnas.
dat1["Fecho"] = dat1["Date"] +" "+ dat1["Time"]
dat1.drop( ["Date","Time"], axis=1, inplace=True )
dat1.head()
dat1.dtypes

dat1['albedo_Avg'] = pd.to_numeric(dat1['albedo_Avg'],errors='coerce')

#Convertimos la variable Fecho a la variable Fecha de tipo datetime
dat1['Fecha'] = pd.to_datetime(dat1.apply(lambda x: x['Fecho'], 1), dayfirst=True)
dat1 = dat1.drop(['Fecho'], 1)
dat1.dtypes

dat1['Año'] = dat1['Fecha'].dt.year
dat1['Mes'] = dat1['Fecha'].dt.month
dat1['Dia'] = dat1['Fecha'].dt.day
dat1['Hora']= dat1['Fecha'].dt.hour
dat1.dtypes

dat1=dat1.drop(['Fecha'],axis=1)
dat1.head()


dat1['RHmax']=np.round(dat1.groupby(['Dia','Mes'])['RH'].transform('max'),decimals=2)
dat1['RHmin']=np.round(dat1.groupby(['Dia','Mes'])['RH'].transform('min'),decimals=2)
dat1['RHmean']=np.round(dat1.groupby(['Mes'])['RH'].transform('mean'),decimals=2)
dat1['RHmax_m']=np.round(dat1.groupby(['Mes'])['RHmax'].transform('mean'),decimals=2)
dat1['RHmin_m']=np.round(dat1.groupby(['Mes'])['RHmin'].transform('mean'),decimals=2)




dat1['Tmax']=np.round(dat1.groupby(['Dia','Mes'])['AirTC_Avg'].transform('max'),decimals=2)
dat1['Tmin']=np.round(dat1.groupby(['Dia','Mes'])['AirTC_Avg'].transform('min'),decimals=2)

dat1['Tmean']=np.round(dat1.groupby(['Mes'])['AirTC_Avg'].transform('mean'),decimals=2)
dat1['Tmax_m']=np.round(dat1.groupby(['Mes'])['Tmax'].transform('mean'),decimals=2)
dat1['Tmin_m']=np.round(dat1.groupby(['Mes'])['Tmin'].transform('mean'),decimals=2)


dat1['Rs']=np.round(dat1.groupby(['Mes'])['Rn_Avg'].transform('mean'),decimals=2)
dat1['Vel_viento']=np.round(dat1.groupby(['Mes'])['WS_ms_S_WVT'].transform('mean'),decimals=2)


dat1['albedo']=np.round(dat1.groupby(['Mes'])['albedo_Avg'].transform('mean'),decimals=2)


#Quitamos datos repetidos por mes y reseteamos el índice.
dat1 = dat1.drop_duplicates(subset=['Mes'])
dat1=dat1.reset_index(drop=True)
dat1.head()
len(dat1)
dat1

dat1=dat1.drop(['Dia','Hora','AirTC_Avg','RH','Rn_Avg','WS_ms_S_WVT',\
                'albedo_Avg','Año','RHmax','RHmin','Tmin','Tmax'],axis=1)

dat1

#dat1=dat1.drop_duplicates(subset=["Vel_viento",\
 #                       'Tmean','Tmax_m',"Tmin_m",\
  #                      "RHmean","RHmin_m",'RHmax_m',\
   #                     "Rs"])

dat1.head()

dat1=dat1.drop(12)
#Reiniciamos el índice y agregamos los meses
mes=["Enero",'Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
dat1['Mes']=mes
dat1.set_index('Mes')
dat1

#

#Gráfica de Temperaturas, RH y Rs
X = mes                 
N = np.arange(12)         
Y1 = dat1['Tmean']     
Y2 = dat1['Tmax_m']     
Y3 = dat1['Tmin_m']     


plt.plot(Y1, label = 'Temperatura prom', color = '#D2691E')   
plt.plot(Y2, label = 'Temperatura min', color = '#1E90FF')   
plt.plot(Y3, label = 'Temperatura max', color = '#C71585')   

plt.xticks(N, X, size = 'small', color = 'k', rotation = 90)
plt.xlabel("Mes")   
plt.ylabel("Temperatura (ºC)")  
plt.legend()
plt.grid()
plt.title('Temperaturas mensuales en 2019')
plt.savefig('Fig_1',plt=2000)
plt.show()

#############################################################


X = mes                 
N = np.arange(12)         
Y1 = dat1['RHmean']     
Y2 = dat1['RHmax_m']     
Y3 = dat1['RHmin_m']     


plt.plot(Y1, label = 'RH prom', color = '#D2691E')   
plt.plot(Y2, label = 'RH min', color = '#1E90FF')   
plt.plot(Y3, label = 'RH max', color = '#C71585')   

plt.xticks(N, X, size = 'small', color = 'k', rotation = 90)
plt.xlabel("Mes")   
plt.ylabel("Humedad relativa")  
plt.legend()
plt.grid()
plt.title('RH mensuales en 2019')
plt.savefig('Fig_2',plt=2000)
plt.show()


###########################################################


X = mes                 
N = np.arange(12)         
Y1 = dat1['Rs']     
    


plt.plot(Y1, label = 'Rs', color = '#D2691E')   
  

plt.xticks(N, X, size = 'small', color = 'k', rotation = 90)
plt.xlabel("Mes")   
plt.ylabel("Radiación solar")  
plt.legend()
plt.grid()
plt.title('Rs mensuales en 2019')
plt.savefig('Fig_3',plt=2000)
plt.show()


#####################################################################

#####################################################################

#parte dos:

#ecuación de Jensen and Haise

ETo=[]

for k in range(len(dat1)):
    ETo.append((0.0252*dat1['Tmean'][k]+0.078)*dat1['Rs'][k])
    
# agregamos al data frame ETo
ETo=np.round(ETo,decimals=2)
dat1['ETo']=ETo
dat1

# eliminamos la fila con el indice 13, la última
dat1=dat1.drop(13)
dat1

# Ecuación (31) de Valiantzas (2012)

fi = (math.pi/180)*28.94917
ETo_1=[]

for k in range(len(dat1)):
    ETo_1.append((0.0393*dat1['Rs'][k]*(dat1['Tmean'][k]+9.5)\
                  **0.5-0.19*dat1['Rs'][k]**0.6*fi**0.15+0.0061*\
                  (dat1['Tmean'][k]+20)*(1.12*dat1['Tmean'][k]-\
                  dat1['Tmin_m'][k]-2)**0.7))


ETo_1
ETo_1=np.round(ETo_1,decimals=2)
dat1['ETo_1']=ETo_1
len(dat1)
len(ETo_1)



#############################################################

#############################################################

#Ecuación 34: 

#Para esto, debemos obtener primero Ra. Calcularemos los parámetros de Ra:
dr = []
delta = []
w = []
φ = (math.pi/180)*28.94917
for m in range (0,len(dat1)):
   # α = dat1['albedo'][i]
    j = int(30.4*m - 15)
    dr.append(1 + 0.033*math.cos(((2*math.pi)/365)*j))
    delta.append(0.409*math.sin(((2*math.pi)/365)*j - 1.39))
    w.append(math.acos(-math.tan(φ)*math.tan(0.409*math.sin(((2*math.pi)/365)*j - 1.39))))
    
    
    


#Guardamos los arreglos en un dataframe auxiliar
df_aux = pd.DataFrame()
df_aux['dr'] = dr
df_aux['delta'] = delta
df_aux['w'] = w
df_aux = df_aux.apply(pd.to_numeric, errors='coerce')

φ = (math.pi/180)*28.94917

df_aux.head()



#Calculamos Ra por mes:
Ra = []
for i in range (0,len(dat1)):
    Ra.append(((24*60)/math.pi)*0.0820*df_aux['dr'][i]*(df_aux['w'][i]*math.sin(φ)*\
               math.sin(df_aux['delta'][i]) + math.cos(φ)*math.cos(df_aux['delta'][i])\
               *math.sin(df_aux['w'][i])))
    
    
    
#Convertimos el arreglo de Ra en float64:
df_aux['Ra'] = Ra
#df_aux['Ra'] = df_aux['Ra'].apply(lambda col:pd.to_numeric(col, errors='coerce'))



#Calculamos la ecuación 34:
ET0_34 = [] 
for i in range (0,len(dat1)):
    ET0_34.append(0.051*(1 - dat1['albedo'][i])*\
                  dat1['Rs'][i]*(dat1\
                               ['Tmean'][i] + 9.5)**0.5 - 2.4*\
                               (dat1['Rs'][i]/df_aux['Ra']\
                                [i])**2 + 0.048*(dat1\
                                ['Tmean'][i] + 20)*\
                                (1 - dat1['RHmean']\
                                 [i]/100)*(0.5 + 0.536*dat1\
                                 ['Vel_viento'][i]) + 0.00012*101)
    
    
    
#Creamos un dataframe con los ET0 calculados:
ET0 = {'ET0_7': ET0_7, 'ET0_31': ET0_31, 'ET0_34':ET0_34,'MES':mes}
ET0 = pd.DataFrame(data=ET0)
ET0.set_index('MES')


















