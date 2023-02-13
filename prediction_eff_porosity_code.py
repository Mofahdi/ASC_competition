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
data = pd.read_excel("ASC Student Challenge 2021_Data Set.xlsx"); df=data.iloc[450:,:]
#ASC_2021_data = pd.read_excel("ASC Student Challenge 2021_Data Set.xlsx")
o=df.loc[:, 'Heat Rate 1 [C/min]':'Autoclave Duration [min]']
orig_columns=o.columns
print(max(df['Heat Rate 2 [C/min]']), min(df['Heat Rate 2 [C/min]']))
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
print(df.columns)

x_cols=['Vacuum Pressure (*Patm) _Pa_', 'Cure Cycle Total Time _min_', 
        'Vacuum Duration _min_', 'Ramp 2 Duration _min_',
        'Heat Rate 1 _C/min_', 'Temperature Dwell 2 _min_',
        'Heat Rate 2 _C/min_', 'Temperature Dwell 1 _min_', 'Ramp 1 Duration _min_',
        'Autoclave Pressure (*Patm) _Pa_', 'Autoclave Duration _min_']
x=df.loc[:, x_cols]; print(x); print(x.columns)

y = df.loc[:, 'Eff. Porosity (%)']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, shuffle=True)
#%% Plots for Facesheet Consolidation - Max. Fibre Volume Fraction

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
# =============================================================================
# GUI
# tkinter GUI

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 700, height = 800)
canvas1.pack()


# New_Vacuum_Pressure label and input box
label1 = tk.Label(root, text='Vacuum Pressure (0-1): ')
canvas1.create_window(160, 100, window=label1)
entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(380, 100, window=entry1)

# New_AutoclavePressure_Rate label and input box
label2 = tk.Label(root, text=' Cure Cycle Total Time [min] (200-337): ')
canvas1.create_window(160, 125, window=label2)
entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 125, window=entry2)

# New_AutoclavePressure_Rate label and input box
label3 = tk.Label(root, text=' Vaccuum Duration (120-336): ')
canvas1.create_window(160, 150, window=label3)
entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 150, window=entry3)

label4 = tk.Label(root, text='Ramp 2 Duration [min] (0-67): ')
canvas1.create_window(160, 175, window=label4)
entry4 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 175, window=entry4)

label5 = tk.Label(root, text='Heat Rate 1 [C/min] (1-4): ')
canvas1.create_window(160, 200, window=label5)
entry5 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 200, window=entry5)

label6 = tk.Label(root, text='Temperature Dwell 2 [min] (0-120): ')
canvas1.create_window(160, 225, window=label6)
entry6 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 225, window=entry6)

label7 = tk.Label(root, text='Heat Rate 2 [C/min] (0-4): ')
canvas1.create_window(160, 250, window=label7)
entry7 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 250, window=entry7)

label8 = tk.Label(root, text='Temperature Dwell 1 [min] (60-120): ')
canvas1.create_window(160, 275, window=label8)
entry8 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 275, window=entry8)

label9 = tk.Label(root, text='Ramp 1 Duration _min_ (22.5-90): ')
canvas1.create_window(160, 300, window=label9)
entry9 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 300, window=entry9)

label10 = tk.Label(root, text='Autoclave Pressure (*Patm) [Pa] (2-4): ')
canvas1.create_window(160, 325, window=label10)
entry10 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 325, window=entry10)

label11 = tk.Label(root, text='Autoclave Duration [min] (120-336): ')
canvas1.create_window(160, 350, window=label11)
entry11 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(380, 350, window=entry11)

# Cycle Number : min: 1 max: 1800
# Heat Rate 1 _C/min_ : min: 1 max: 4
# Ramp 1 Duration _min_ : min: 22.5 max: 90.0
# Temperature Dwell 1 _min_ : min: 60 max: 120
# Heat Rate 2 _C/min_ : min: 0 max: 4
# Ramp 2 Duration _min_ : min: 0.0 max: 67.0
# Temperature Dwell 2 _min_ : min: 0 max: 120
# Vacuum Pressure (*Patm) _Pa_ : min: 0.01 max: 1.0
# Vacuum Start Time _min_ : min: 1 max: 80
# Vacuum Duration _min_ : min: 120.0 max: 336.0
# Autoclave Pressure (*Patm) _Pa_ : min: 2 max: 4
# Cure Cycle Total Time _min_ : min: 200.0 max: 337.0
# Autoclave Start Time _min_ : min: 1 max: 80
# Autoclave Duration _min_ : min: 120.0 max: 336.0

def values(): 
    global New_Vacuum_Pressure #our 1st input variable
    New_Vacuum_Pressure = float(entry1.get()) 
    
    global New_cure_cycle #our 2nd input variable
    New_cure_cycle = float(entry2.get()) 
    
    global New_vacuum_duration #our 2nd input variable
    New_vacuum_duration = float(entry3.get()) 
    
    global New_ramp2_duration
    New_ramp2_duration=float(entry4.get())
    
    global New_heat_rate1
    New_heat_rate1=float(entry5.get())

    global New_temp_dwell_2
    New_temp_dwell_2=float(entry6.get())
    
    global New_heat_rate2
    New_heat_rate2=float(entry7.get())
    
    global New_temp_dwell_1
    New_temp_dwell_1=float(entry8.get())
    
    global New_ramp1_duration
    New_ramp1_duration=float(entry9.get())
    
    global New_autoclave_press
    New_autoclave_press=float(entry10.get())
    
    global New_autoclave_duration
    New_autoclave_duration=float(entry11.get())

    new_data={'Vacuum Pressure (*Patm) _Pa_': New_Vacuum_Pressure,
              'Cure Cycle Total Time _min_': New_cure_cycle,
              'Vacuum Duration _min_': New_vacuum_duration,
              'Ramp 2 Duration _min_': New_ramp2_duration,
              'Heat Rate 1 _C/min_':New_heat_rate1,
              'Temperature Dwell 2 _min_':New_temp_dwell_2,
              'Heat Rate 2 _C/min_':New_heat_rate2,
              'Temperature Dwell 1 _min_':New_temp_dwell_1,
              'Ramp 1 Duration _min_':New_ramp1_duration,
              'Autoclave Pressure (*Patm) _Pa_':New_autoclave_press,
              'Autoclave Duration _min_':New_autoclave_duration}
    new_data_df=pd.DataFrame([new_data])
    Prediction_result  = ('Predicted Effective Porosity: ', regr.predict(new_data_df))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(290, 420, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict Eff. Porosity',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(290, 385, window=button1)
 
#plot 1st scatter 
# figure3 = plt.Figure(figsize=(5,4), dpi=100)
# ax3 = figure3.add_subplot(111)
# ax3.scatter(ASC_2021_data['Autoclave Pressure (*Patm) [Pa]'].astype(float),ASC_2021_data['Max (Fiber Volume Fraction) (%)'].astype(float), color = 'r')
# scatter3 = FigureCanvasTkAgg(figure3, root) 
# scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
# ax3.set_title('Max-Vf Vs. Autoclave Pressure')
# ax3.set_xlabel('Autoclave Pressure (*Patm) [Pa]')
# ax3.set_ylabel('Max (Fiber Volume Fraction) (%)')


# #plot 2nd scatter 
# figure4 = plt.Figure(figsize=(5,4), dpi=100)
# ax4 = figure4.add_subplot(111)
# ax4.scatter(ASC_2021_data['Vacuum Pressure (*Patm) [Pa]'].astype(float),ASC_2021_data['Max (Fiber Volume Fraction) (%)'].astype(float), color = 'g')
# scatter4 = FigureCanvasTkAgg(figure4, root) 
# scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
# ax4.set_xlabel('Vacuum Pressure (*Patm) [Pa]')
# ax4.set_title('Max-Vf Vs. Vacuum Pressure')
# ax4.set_ylabel('Max (Fiber Volume Fraction) (%)')

root.mainloop()


#%%
