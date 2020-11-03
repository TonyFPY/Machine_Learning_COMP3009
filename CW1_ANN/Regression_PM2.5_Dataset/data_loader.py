import numpy as np
import pandas as pd
import datetime

def PM2Dot5():
    path = '.../PRSA_data_2010.1.1-2014.12.31.csv'
    # names = ['No','year','month','day','hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']
    df = pd.read_csv(path,index_col=False)

    # drop the rows directly -> mess up the order
    # first 24 rows have pm2.5 value that is NaN -> discard
    # else: forward filling
    df = df[24:].fillna(method='ffill')

    df['time'] = df.apply(lambda x : datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']), axis=1)
    df.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)
    df = df.set_index('time')

    # one-hot encoding for attribute cbwd
    df = df.join(pd.get_dummies(df['cbwd']))
    del df['cbwd']


    X = df.iloc[:,1:].astype(np.float64)
    y = df['pm2.5'].astype(np.float64)

    return X, y
