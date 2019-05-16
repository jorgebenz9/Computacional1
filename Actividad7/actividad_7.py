# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:30:06 2019

@author: olguin aguilar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


dat = pd.read_csv("meteo_nogal_09.csv", engine="python")
dat.head()
