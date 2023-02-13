# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 07:22:02 2021

@author: RAGHU
"""

import IPython as IP 
IP.get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn as sk
from sklearn import datasets, linear_model, multiclass,model_selection, preprocessing, feature_selection, ensemble,metrics,decomposition
import time as time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import lime as lm
# from lime import lime_tabular
# changing color parameters
# cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.close('all')

# Load Data
ASC_2021_data = pd.read_csv("ASC-2021-1.csv")
ASC_2021_data_1=ASC_2021_data.loc[0:449,:]
columns_X=['Heat Rate 1 [C/min]','Ramp 1 Duration [min]','Temperature Dwell 1 [min]','Heat Rate 2 [C/min]','Ramp 2 Duration [min]','Temperature Dwell 2 [min]','Vacuum Pressure (*Patm) [Pa]','Vacuum Start Time [min]','Vacuum Duration [min]','Autoclave Pressure (*Patm) [Pa]','Autoclave Start Time [min]','Autoclave Duration [min]']
columns_y=['AD. Porosity (%)','PR. Porosity (%)','Eff. Porosity (%)','Max (Fiber Volume Fraction) (%)','Cure Cycle Total Time [min]','PR. Volume [m^3]','Class']
ASC_2021_data_X = ASC_2021_data_1[columns_X]
ASC_2021_data_y = ASC_2021_data_1[columns_y]
X = ASC_2021_data_X
y = ASC_2021_data_y['AD. Porosity (%)']
X=columns_X
Y=['Class']
y=ASC_2021_data[Y]
#%% Plots for Facesheet Consolidation - Max. Fibre Volume Fraction
#plot the figure
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_X['Vacuum Pressure (*Patm) [Pa]'], ASC_2021_data_y['AD. Porosity (%)'],color='red')
plt.title('Adhesive Porosity vs Vaccuum Pressure - with 1 Heat Ramp', fontsize=14)
plt.xlabel('Vacuum Pressure (*Patm) [Pa]', fontsize=14)
plt.ylabel('AD. Porosity (%)', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()
#plot the figure for Vf
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_X['Autoclave Pressure (*Patm) [Pa]'],ASC_2021_data_y['Max (Fiber Volume Fraction) (%)'], color='blue')
plt.title('Max. Fibre Volume Fraction vs Autoclave Pressure - with 1 Heat Ramp', fontsize=14)
plt.ylabel('Max (Fiber Volume Fraction) (%)', fontsize=14)
plt.xlabel('Autoclave Pressure (*Patm) [Pa]', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#plot the figure for Vf
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_y['PR. Volume [m^3]'],ASC_2021_data_y['Max (Fiber Volume Fraction) (%)'], marker = 's',color='blue')
plt.title('Max. Fibre Volume Fraction vs PR Volume - with 1 Heat Ramp', fontsize=14)
plt.ylabel('Max (Fiber Volume Fraction) (%)', fontsize=14)
plt.xlabel('PR. Volume [m^3]', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#plot the figure for Vf
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_X['Vacuum Pressure (*Patm) [Pa]'],ASC_2021_data_y['Max (Fiber Volume Fraction) (%)'], marker = 's',color='blue')
plt.title('Max. Fibre Volume Fraction vs Vaccuum Pressure - with 1 Heat Ramp', fontsize=14)
plt.ylabel('Max (Fiber Volume Fraction) (%)', fontsize=14)
plt.xlabel('Vacuum Pressure (*Patm) [Pa]', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#plot the figure for Effective Porosity
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_y['PR. Volume [m^3]'],ASC_2021_data_y['Eff. Porosity (%)'], marker = 's',color='blue')
plt.title('Effective porosity vs PR Volume - with 1 Heat Ramp', fontsize=14)
plt.ylabel('Eff. Porosity (%)', fontsize=14)
plt.xlabel('PR. Volume [m^3]', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()


#plot the figure for Vac vs Aut Pressure
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_X['Vacuum Pressure (*Patm) [Pa]'],ASC_2021_data_X['Autoclave Pressure (*Patm) [Pa]'], marker = 's',color='blue')
plt.title('Effective porosity vs PR Volume - with 1 Heat Ramp', fontsize=14)
plt.ylabel('Vacuum Pressure (*Patm) [Pa]', fontsize=14)
plt.xlabel('Autoclave Pressure (*Patm) [Pa]', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#3D plot figure
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(ASC_2021_data_X['Autoclave Pressure (*Patm) [Pa]'], ASC_2021_data_X['Vacuum Pressure (*Patm) [Pa]'], ASC_2021_data_y['Max (Fiber Volume Fraction) (%)'])
ax.set_xlabel('Autoclave Pressure'); ax.set_ylabel('Vaccuum Pressure'); ax.set_zlabel('Max Fiber Volume Fraction')
plt.title('Max. Fibre Volume Fraction vs Autoclave Pressure and Vaccuum Pressure - with 1 Heat Ramp', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#%% Plots for Facesheet Consolidation - Prepreg Porosity
#plot the figure - Vac  Pressure vs PR porosity
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_X['Vacuum Pressure (*Patm) [Pa]'], ASC_2021_data_y['PR. Porosity (%)'],color='red')
plt.title('Prepreg Porosity vs Vaccuum Pressure - with 1 Heat Ramp', fontsize=14)
plt.xlabel('Vacuum Pressure (*Patm) [Pa]', fontsize=14)
plt.ylabel('PR. Porosity (%)', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()
#plot the figure for AD Porosity vs PR porosity
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_y['AD. Porosity (%)'],ASC_2021_data_y['PR. Porosity (%)'], color='blue')
plt.title('Prepreg Porosity vs Adhesive Porosity - with 1 Heat Ramp', fontsize=14)
plt.ylabel('PR. Porosity (%)', fontsize=14)
plt.xlabel('AD. Porosity (%)', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#plot the figure for Vf vs PR porosity
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_y['PR. Porosity (%)'],ASC_2021_data_y['Max (Fiber Volume Fraction) (%)'], marker = 's',color='blue')
plt.title('Max. Fibre Volume Fraction vs PR Porosity - with 1 Heat Ramp', fontsize=14)
plt.ylabel('Max (Fiber Volume Fraction) (%)', fontsize=14)
plt.xlabel('PR. Porosity (%)', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#plot the figure for Prepreg Porosity vs prepreg volume
plt.figure(figsize=(6.5,4))
plt.scatter( ASC_2021_data_y['PR. Volume [m^3]'],ASC_2021_data_y['PR. Porosity (%)'], marker = 's',color='blue')
plt.title('PR Porosity vs PR Volume - with 1 Heat Ramp', fontsize=14)
plt.ylabel('PR. Porosity (%)', fontsize=14)
plt.xlabel('PR. Volume [m^3]', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()


#3D plot figure - prepreg porosity
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(ASC_2021_data_y['Eff. Porosity (%)'], ASC_2021_data_y['AD. Porosity (%)'],ASC_2021_data_y['PR. Porosity (%)'])
ax.set_xlabel('Eff. Porosity (%)'); ax.set_ylabel('AD. Porosity (%)'); ax.set_zlabel('PR. Porosity (%)')
plt.title('PR Porosity vs AD Porosity and Eff. Porosity - with 1 Heat Ramp', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

#3D plot figure - Final Facesheet consolidation
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(ASC_2021_data_y['Max (Fiber Volume Fraction) (%)'], ASC_2021_data_y['PR. Porosity (%)'],ASC_2021_data_y['PR. Volume [m^3]'])
ax.set_xlabel('Max (Fiber Volume Fraction) (%)'); ax.set_ylabel('PR. Porosity (%)'); ax.set_zlabel('PR. Volume [m^3]')
plt.title('PR Porosity vs Max Fiber Volume Fraction and PR Volume - with 1 Heat Ramp', fontsize=14)
plt.grid(True)
plt.show()
plt.tight_layout()

# =============================================================================
# Multiple Linear Regression
# =============================================================================


# X=ASC_2021_data_1[['Heat Rate 1 [C/min]','Ramp 1 Duration [min]','Temperature Dwell 1 [min]','Vacuum Pressure (*Patm) [Pa]','Vacuum Start Time [min]','Vacuum Duration [min]','Autoclave Pressure (*Patm) [Pa]','Autoclave Start Time [min]','Autoclave Duration [min]']].astype(float)
X = ASC_2021_data_1[['Vacuum Pressure (*Patm) [Pa]','Autoclave Pressure (*Patm) [Pa]','Autoclave Duration [min]']].astype(float) # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
y = ASC_2021_data_1['Max (Fiber Volume Fraction) (%)'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,shuffle=True)


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

coeff_df = pd.DataFrame(regr.coef_, X.columns, columns=['Coefficient'])

y_pred=regr.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
scores=regr.score(X_test,y_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# prediction with sklearn
# Vacuum_Pressure = 0.01
# Autoclave_Pressure = 3.5
# print ('Predicted Max Vol Fraction: \n', regr.predict([[Vacuum_Pressure,Autoclave_Pressure]]))


X = sm.add_constant(X_train) # adding a constant
 
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) 
 
print_model = model.summary()
print(print_model)


# =============================================================================
# GUI
# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 600, height = 300)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# New_Vacuum_Pressure label and input box
label1 = tk.Label(root, text='Type Vacuum Pressure (0-1): ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_AutoclavePressure_Rate label and input box
label2 = tk.Label(root, text=' Type Autoclave Pressure(2-4): ')
canvas1.create_window(100, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

# New_AutoclavePressure_Rate label and input box
label3 = tk.Label(root, text=' Type Autoclave Duration(120-360): ')
canvas1.create_window(100, 140, window=label3)

entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 140, window=entry3)
def values(): 
    global New_Vacuum_Pressure #our 1st input variable
    New_Vacuum_Pressure = float(entry1.get()) 
    
    global New_Autoclave_Pressure #our 2nd input variable
    New_Autoclave_Pressure = float(entry2.get()) 
    
    global New_Autoclave_Duration #our 2nd input variable
    New_Autoclave_Duration = float(entry3.get()) 
    
    Prediction_result  = ('Predicted Max Fibre Vol. Fraction: ', regr.predict([[New_Vacuum_Pressure,New_Autoclave_Pressure,New_Autoclave_Duration]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict Max Fibre Vol. Fraction',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 160, window=button1)
 
#plot 1st scatter 
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(ASC_2021_data_1['Autoclave Pressure (*Patm) [Pa]'].astype(float),ASC_2021_data_1['Max (Fiber Volume Fraction) (%)'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.set_title('Max-Vf Vs. Autoclave Pressure')
ax3.set_xlabel('Autoclave Pressure (*Patm) [Pa]')
ax3.set_ylabel('Max (Fiber Volume Fraction) (%)')




#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(ASC_2021_data_1['Vacuum Pressure (*Patm) [Pa]'].astype(float),ASC_2021_data_1['Max (Fiber Volume Fraction) (%)'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.set_xlabel('Vacuum Pressure (*Patm) [Pa]')
ax4.set_title('Max-Vf Vs. Vacuum Pressure')
ax4.set_ylabel('Max (Fiber Volume Fraction) (%)')

root.mainloop()
# =============================================================================




















