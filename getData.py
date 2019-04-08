import pandas as pd
import numpy as np
import random

def prepareData(location):
    df = pd.read_csv("weatherAUS.csv",sep=',',header=0, names=None)

    df_filter = df[df['Location'] == location]
    df_filter = df_filter.drop(columns = ['Location','RISK_MM'])

    rows, cols = df_filter.shape
    for col,size in df_filter.count().iteritems():
        if size < (rows/2) :
            df_filter = df_filter.drop(columns = [col])
    
    df_filter = df_filter.dropna(how = 'any')

    df_filter['Date'] = pd.to_datetime(df_filter['Date'])
    temp = np.array(['Ene-Mar','Abr-Jun','Jul-Sep','Oct-Dic'])
    df_filter['Date'] = temp[np.ceil(df_filter['Date'].dt.month/3.0).astype(int) - 1]
    
    #Remember add Date again
    df_filter = pd.get_dummies(df_filter, columns=['Date','WindDir3pm','WindDir9am','WindGustDir'])
    df_filter['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
    df_filter['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

    rt = df_filter.pop('RainTomorrow')
    rows, cols = df_filter.shape
    df_filter.insert(cols   ,'RainTomorrow',rt)
    df_filter = df_filter.reset_index(drop=True)

    for i in df_filter.columns:
        if i == 'RainToday':
            break
        df_filter[i] = (df_filter[i] - df_filter[i].mean())/df_filter[i].std()

    with open("mi_data_" + location + ".csv",mode='w',newline='') as f:
        df_filter.to_csv(f)
    return df_filter

def separateDataTrainTest(data, train):
    df_train = pd.DataFrame(columns = data.columns)
    df_test = pd.DataFrame(columns = data.columns)
    train = train/100.0
    for i in range(0,len(data.index)):
        rn = random.random()
        if rn <= train:
            df_train = df_train.append(data.iloc[[i]])
        else:
            df_test = df_test.append(data.iloc[[i]])
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    with open("mi_train.csv",mode='w',newline='') as f:
        df_train.to_csv(f)
    with open("mi_test.csv",mode='w',newline='') as f:
        df_test.to_csv(f)
    return df_train,df_test

def separateDataXY(train,test):
    xtrain = train.iloc[:,:len(train.columns)-1].to_numpy(dtype = np.float32)
    ytrain = train.iloc[:,-1].to_numpy(dtype = np.float32)

    xtest = test.iloc[:,:len(test.columns)-1].to_numpy(dtype = np.float32)
    ytest = test.iloc[:,-1].to_numpy(dtype = np.float32)

    return xtrain,ytrain,xtest,ytest





