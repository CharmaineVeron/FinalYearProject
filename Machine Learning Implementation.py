from turtle import color
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import statistics
   
# Loading dataset
dataset = pd.read_csv(r'C:\Users\user\Documents\FYP\TelecomData10-18pm.csv')

models = [LinearRegression(),   SVR(kernel='rbf',C=50),MLPRegressor(solver='lbfgs', learning_rate_init = 0.18,momentum=0.18,hidden_layer_sizes=4, max_iter=10000),BayesianRidge()]
model_acr = ['LR','SVR','NN','BRR']

#MAE, RMSE, MAPE results
res =[[],[],[],[]]
res1 =[[],[],[],[]]
res2 =[[],[],[],[]]

for count in range(len(model_acr)):
    # Repeat experiment for each window size 3 to 10
    for i in range(8):
        mae = []
        mape = []
        rmse = []
        # Repeat experiment 10 times for each model
        for a in range(10):
            y = dataset.iloc[:,-1].values
            X  = dataset.iloc[:,7-i:-1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
            
            model = models[count]
            model.fit(X_train, y_train)
            

            prediction = model.predict(X_test)


            mae.append(round(mean_absolute_error(y_test, prediction),2))
            rmse.append(round(mean_squared_error(y_test, prediction, squared=False),2))
            mape.append(round(np.mean(np.abs((y_test - prediction) / y_test)) * 100,2))

  
        res[count].append(round(statistics.mean(mae),2))
        res1[count].append(round(statistics.mean(rmse),2))
        res2[count].append(round(statistics.mean(mape),2))

for x, y, z, a in zip(*res):
    print(x, y, z, a)
print("\n")
for x, y, z, a in zip(*res1):
    print(x, y, z, a)
print("\n")
for x, y, z, a in zip(*res2):
    print(x, y, z, a)


