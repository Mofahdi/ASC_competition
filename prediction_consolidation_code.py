# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:49:49 2021

@author: Mohammed Al-Fahdi
"""

import IPython as IP 
IP.get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import xgboost as xgb

import tkinter as tk


plt.close('all')

# Load Data
df = pd.read_excel("ASC Student Challenge 2021_Data Set.xlsx"); df=df.iloc[450:,:]
ASC_2021_data = pd.read_excel("ASC Student Challenge 2021_Data Set.xlsx")
o=df.loc[:, 'Heat Rate 1 [C/min]':'Autoclave Duration [min]']
orig_columns=o.columns
print(max(df['Heat Rate 2 [C/min]']), min(df['Heat Rate 2 [C/min]']))

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
print(df.columns)

x=df.loc[:, ['Vacuum Pressure (*Patm) _Pa_', 'Autoclave Pressure (*Patm) _Pa_', 'Autoclave Duration _min_', 'Vacuum Duration _min_']];

y = df.loc[:, 'Max (Fiber Volume Fraction) (%)']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, shuffle=True)

#%% Machine learning model 

# =============================================================================
# Gradient Boosting Regression
# =============================================================================
params = {'n_estimators': 800}
regr = xgb.XGBRegressor(max_depth=4, learning_rate=0.03, n_estimators=params['n_estimators'], verbosity=1, objective='reg:squarederror',
 	booster='gbtree', tree_method='auto', n_jobs=1, gamma=0.0001, min_child_weight=8,max_delta_step=0,
 	subsample=0.6, colsample_bytree=0.7, colsample_bynode=1, reg_alpha=0,
 	reg_lambda=4, scale_pos_weight=1, base_score=0.6, missing=None,
 	num_parallel_tree=1, importance_type='gain', eval_metric='mae',nthread=4).fit(x_train,y_train)


#%%

y_pred=regr.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
scores=regr.score(x_test,y_test)
r2_test=r2_score(y_test, y_pred); 
print('R2 score:',r2_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


#%%

new_data={'Vacuum Pressure (*Patm) _Pa_': 0,
          'Autoclave Pressure (*Patm) _Pa_': 5,
          'Autoclave Duration _min_': 120,
          'Vacuum Duration _min_':120}; 

new=pd.DataFrame.from_dict([new_data]); 
prediction=regr.predict(new)
print(prediction)
#%%
# =============================================================================
# GUI
# tkinter GUI

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 700, height = 400)
canvas1.pack()


# New_Vacuum_Pressure label and input box
label1 = tk.Label(root, text='Type Vacuum Pressure (0.01-1): ')
canvas1.create_window(120, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(320, 100, window=entry1)

# New_AutoclavePressure_Rate label and input box
label2 = tk.Label(root, text=' Type Autoclave Pressure(2-4): ')
canvas1.create_window(120, 130, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(320, 130, window=entry2)

# New_AutoclavePressure_Rate label and input box
label3 = tk.Label(root, text=' Type Autoclave Duration(120-336): ')
canvas1.create_window(120, 160, window=label3)

entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(320, 160, window=entry3)

label4 = tk.Label(root, text=' Type Vaccuum Duration(120-336): ')
canvas1.create_window(120, 190, window=label4)

entry4 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(320, 190, window=entry4)

def values(): 
    global New_Vacuum_Pressure #our 1st input variable
    New_Vacuum_Pressure = float(entry1.get()) 
    
    global New_Autoclave_Pressure #our 2nd input variable
    New_Autoclave_Pressure = float(entry2.get()) 
    
    global New_Autoclave_Duration #our 2nd input variable
    New_Autoclave_Duration = float(entry3.get()) 
    
    global New_Vacuum_Duration
    New_Vacuum_Duration=float(entry4.get())
    
    new_data={'Vacuum Pressure (*Patm) _Pa_': New_Vacuum_Pressure,
              'Autoclave Pressure (*Patm) _Pa_': New_Autoclave_Pressure,
              'Autoclave Duration _min_': New_Autoclave_Duration,
              'Vacuum Duration _min_': New_Vacuum_Duration}
    new_data_df=pd.DataFrame([new_data])
    Prediction_result  = ('Predicted Max Fibre Vol. Fraction: ', regr.predict(new_data_df))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(290, 300, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict Max Fibre Vol. Fraction',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(290, 230, window=button1)
 

root.mainloop()


#%%
