# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:11:06 2018

@author: olguin aguilar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Descargar el archivo
data = pd.read_csv("Datos_Hermosillo_editado.txt",sep="\s+",skipfooter=1,skiprows=1,engine='python',names=['Fecha', 'Precip', 'Evap', 'Tmax', 'Tmin'])
data = data.replace('Nulo','NA')
#Cambiamos el tipo de variable de las columnas a flotante
data[['Precip','Evap','Tmax','Tmin']] = data[['Precip','Evap','Tmax','Tmin']] \
.apply(pd.to_numeric, errors='coerce')
data['NFecha'] =  pd.to_datetime(data['Fecha'], format='%d/%m/%Y')
data = data.drop('Fecha', 1)
# Crear columnas con Año y Mes extraídos de la fecha 
data['Año'] = data['NFecha'].dt.year
data['Mes'] = data['NFecha'].dt.month
# Número de años distintos data['Año'].unique(),
NumA = len(data['Año'].unique())

######################################################################


#utilizando un loop
total=0.0
#total=[]
for i in range(12):
    PrecipMensual = data['Precip'][data['Mes']==[i+1]].sum()/NumA
    
    total=total+PrecipMensual

    print("Mes", i+1,":", np.round(PrecipMensual, decimals=2),\
          "mm", ", Acumulada:", 
          np.round(total, decimals=2), "mm")


datos=[7.51,12.68,14.42,15.09,15.38,16.59,29.94,57.75,73.93,81.91,85.48,98.77]
meses = ['Mes 1','Mes 2','Mes 3','Mes 4','Mes 5','Mes 6','Mes 7','Mes 8',\
         'Mes 9','Mes 10','Mes 11','Mes 12']
xx = np.arange(1,len(datos)+1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(xx,datos,width=0.5,color=(0,1,0),align='center')
ax.set_xticks(xx)
ax.set_xticklabels(meses)
ax.set_title("Precipitación mensual acumulada")
ax.set_ylabel('Precipitación(mm)')

plt.show

###########################################################################

# Años húmedos

dat=[]
for i in range(1973,2011):
    
    PrecipAnual = data['Precip'][data['Año']==[i+1]].sum()
    dat.append(np.round(PrecipAnual,decimals=0))

print(dat)

# Años húmedos acumulados

dats=[]
total=0
for k in range(1973,2011):
    #total=0
    PrecipAnual = data['Precip'][data['Año']==[i+1]].sum()
    total = total +  PrecipAnual
    dats.append(np.round(total, decimals=0))
#    print(total)

print(dats)
    

anos =[]
for i in range(1973,2011):
    anos.append(i+1)

print(anos)



anos=['1974','1975','1976','1977','1978','1979','1980','1981',\
    '1982','1983','1984','1985','1986','1987','1988','1989','1990','1991',\
   '1992','1993','1994','1995', '1996', '1997', '1998', '1999', '2000',\
   '2001', '2002','2003', '2004', '2005', '2006','2007','2008','2009',\
   '2010','2011']




yy = np.arange(1,len(dats)+1)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(yy,dats,width=0.5,color=(0,1,0),align='center')
ax.set_xticks(yy)
ax.set_xticklabels(anos)
ax.set_title("Precipitación acumulada por año")
ax.set_ylabel('Precipitación(mm)')
plt.show()  

#################################################################

# Elabora una gráfica de la evolución de la temperatura máxima y mínima en la
# misma figura, como función del tiempo de la colección de datos. 


tmin = []
for i in range(12):
    TminPromMensual = data[data['Mes']==i+1]['Tmin'].sum()/data[data['Mes']\
                          ==i+1]['Tmin'].count()
    tmin.append(np.round(TminPromMensual, decimals=0))
    
#    print("Tmin Mes", i+1,":", np.round(TminPromMensual, decimals=2), "ºC")

tmax = []
for i in range(12):
    TmaxPromMensual = data[data['Mes']==i+1]['Tmax'].sum()/data[data['Mes']\
                          ==i+1]['Tmax'].count()
    tmax.append(np.round(TmaxPromMensual, decimals=0))
#    print("Tmax Mes", i+1,":", np.round(TmaxPromMensual, decimals=2), "ºC")


aa = np.arange(1,len(tmin)+1)
bb = np.arange(1,len(tmax)+1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(aa,tmin)
ax.plot(bb,tmax)
plt.show()

ax.set_xticks(aa)
ax.set_xticklabels(meses)
ax.set_title("Temperaturas máximas y mínimas por mes")
ax.set_ylabel('Temperaturas (°C)')
ax.set_xlabel('Meses')
ax.grid
plt.show()

######################################################################
# Elabora una gráfica de cajas (boxplot) de la temperatura promedio 
# mensual para la temperatura mínima y máxima por separado.   

tminp = []
for i in range(12):
    TminPromMensual = data[data['Mes']==i+1]['Tmin'].sum()/data[data['Mes']\
                          ==i+1]['Tmin'].count()
    tminp.append(np.round(TminPromMensual, decimals=0))
    
#    print("Tmin Mes", i+1,":", np.round(TminPromMensual, decimals=2), "ºC")

tmaxp = []
for i in range(12):
    TmaxPromMensual = data[data['Mes']==i+1]['Tmax'].sum()/data[data['Mes']\
                          ==i+1]['Tmax'].count()
    tmaxp.append(np.round(TmaxPromMensual, decimals=0))
#    print("Tmax Mes", i+1,":", np.round(TmaxPromMensual, decimals=2), "ºC")


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


ax1.boxplot(tminp)
#ax1.ylabel("temperatura mínima promedio")
#ax1.title("Gráfico de caja y bigotes")
ax1.set_title("Gráfico de caja y bigotes 1")
ax1.set_ylabel('temperatura mínima promedio')

ax2.boxplot(tmaxp)
#ax2.ylabel("temperatura mínima promedio")
#ax2.title("Gráfico de caja y bigotes")
ax2.set_title("Gráfico de caja y bigotes 2")
ax2.set_ylabel('temperatura máxima promedio')

plt.show()

########################################################################

# Elabora una gráfica de cajas de la temperatura mínima y máxima promedio
# anual para cada año por separado. 

tmax2=[]
for i in range(1973,2011):
    NumDatos= data['Tmax'][data['Año']==[i+1]].count()
    TempPromMax = data['Tmax'][data['Año']==[i+1]].sum()/NumDatos
    tmax2.append( np.round(TempPromMax, decimals=0))
    
#    print("Año", i+1,":", np.round(TempPromMax , decimals=2), "°C")

print(tmax2)
    
tmaxdat= [30.0, 28.0, 29.0, 27.0, 28.0, 27.0, 28.0, 27.0, 26.0, 26.0, 27.0,\
          26.0, 27.0, 27.0, 23.0, 27.0, 26.0, 28.0, 27.0, 20.0,\
          29.0, 28.0, 27.0, 26.0]   

# Años más frios
tmin2=[]
for i in range(1973,2011):
    NumDatos= data['Tmin'][data['Año']==[i+1]].count()
    TempPromMin = data['Tmin'][data['Año']==[i+1]].sum()/NumDatos
    tmin2.append( np.round(TempPromMin, decimals=0))
#    print("Año", i+1,":",np.round(TempPromMin , decimals=2), "°C")
    
tmindat=[17.0, 12.0, 12.0, 15.0, 14.0, 13.0, 15.0, 15.0, 14.0, 14.0,\
         14.0, 13.0, 14.0, 16.0, 9.0, 14.0, 13.0, 15.0, 14.0, 5.0,\
         18.0, 16.0, 15.0, 12.0]

fig = plt.figure()
ax3 = fig.add_subplot(121)
ax4 = fig.add_subplot(122)


ax3.boxplot(tmindat)
#ax1.ylabel("temperatura mínima promedio")
#ax1.title("Gráfico de caja y bigotes")
ax3.set_title("Gráfico de caja y bigotes 1")
ax3.set_ylabel('temperatura mínima promedio')
plt.show()

ax4.boxplot(tmaxdat)
#ax2.ylabel("temperatura mínima promedio")
#ax2.title("Gráfico de caja y bigotes")
ax4.set_title("Gráfico de caja y bigotes 2")
ax4.set_ylabel('temperatura máxima promedio')

plt.show()





