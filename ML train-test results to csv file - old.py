from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os



def evaluate(model, i):
    # Loading dataset
    dataset = pd.read_csv(r'C:\Users\user\Documents\FYP\TelecomData00-04am.csv')
 
    
    y = dataset.iloc[:,-1].values
    y = y.reshape(-1,1)
    X = dataset.iloc[:,7-i:-1].values

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    model.fit(X_train,y_train)


    #inverse_transform() method in StandardScaler class to inverse values for prediction
    prediction = model.predict(sc_X.transform(X_test))
    prediction = sc_y.inverse_transform(prediction)


    r2 = r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction, squared=False)
    mape = np.mean(np.abs((y_test - prediction) / y_test)) * 100

    return(r2, mae, rmse, mape)


def main():
    sheet = ['00-04']
    models = [LinearRegression(), SVR(kernel='rbf'),MLPRegressor(solver = 'lbfgs', learning_rate='adaptive', learning_rate_init = 0.38,momentum=0.2,hidden_layer_sizes=4, max_iter=10000),BayesianRidge()]
    model_acr = ['LR','SVR','NN','BRR']

    evaluate()
    for i in sheet:
        out_path = "C:\\Users\\user\\Documents\\FYP\\00-04 results.xlsx"
        df_excel = pd.read_excel(r'C:\Users\user\Documents\FYP\00-04 results.xlsx',  engine='openpyxl') # skiprows=lambda x: x in  [1, 2, 3, 4,5,6,7,8,9,10,11,12,13, 15]
        df_excel = pd.concat([df_excel,pd.DataFrame({" "})], axis=1)
        writer = pd.ExcelWriter(out_path , engine='xlsxwriter')
        df_excel.to_excel(writer)
        writer.save()
        
        for count in range(len(model_acr)):
            r2=[]
            mae = []
            rmse = []
            mape = []   
            print(model_acr[count])    

            model = models[count]
            for b in range(10):
                
                svrr2, svrmae, svrrmse, svrmape = evaluate(model, i)
                r2.append(svrr2)
                mae.append(svrmae)
                rmse.append(svrrmse)
                mape.append(svrmape)
            
            df = pd.DataFrame({'{}_R2'.format(model_acr[count]):r2, '{}_MAE'.format(model_acr[count]):mae, '{}_RMSE'.format(model_acr[count]):rmse, '{}_MAPE'.format(model_acr[count]):mape})
            
            out_path = "C:\\Users\\user\\Documents\\FYP\\00-04 results.xlsx"
            df_excel = pd.read_excel(r'C:\Users\user\Documents\FYP\00-04 results.xlsx',  engine='openpyxl') # skiprows=lambda x: x in  [1, 2, 3, 4,5,6,7,8,9,10,11,12,13, 15]
            # df_excel = pd.concat([df_excel,pd.DataFrame({" "})], axis=1)
            result = pd.concat([df_excel,df], axis=1)

            writer = pd.ExcelWriter(out_path , engine='xlsxwriter')
            result.to_excel(writer)
            writer.save()
            writer.close()

if __name__== "__main__":
    main()