import numpy as np
import pandas as pd

def preprocess_df(train, clust_to_num):
    '''
    train : train dataframe
    clust_to_num : dict containing cluster as a key, according buildings as values
    '''    
    # outlier processing
    train['datetime'] = pd.to_datetime(train['datetime'])
    idx = train[(train.num == 31)&(train.target < 3000)].index[0]
    train.iloc[idx,2] = train.iloc[idx-1,2]/2 + train.iloc[idx+1,2]/2
    idx = train[(train.num == 33)&(train.target < 2000)].index[0]
    train.iloc[idx,2] = train.iloc[idx-1,2]/2 + train.iloc[idx+1,2]/2

    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i+1)]-26))
            else:
                ys.append(np.sum(xs[(i-11):(i+1)]-26))
        return np.array(ys)

    X_train = train.copy()

    # adding datetime features
    X_train['datetime'] = pd.to_datetime(X_train['datetime'])
    X_train['hour'] = X_train['datetime'].dt.hour
    X_train['month'] = X_train['datetime'].dt.month
    X_train['day'] = X_train['datetime'].dt.day
    X_train['date'] = X_train['datetime'].dt.date
    X_train['weekday'] = X_train['datetime'].dt.weekday

    # feature engineering universally applied to every building

    ## cyclic transformation on hour
    X_train['hour_sin'] = np.sin(2 * np.pi * X_train['hour']/23.0)
    X_train['hour_cos'] = np.cos(2 * np.pi * X_train['hour']/23.0)
    ## cyclic transformation on date 
    X_train['date_sin'] = -np.sin(2 * np.pi * (X_train['month']+X_train['day']/31)/12)
    X_train['date_cos'] = -np.cos(2 * np.pi * (X_train['month']+X_train['day']/31)/12)
    ## cyclic transformation on month
    X_train['month_sin'] = -np.sin(2 * np.pi * X_train['month']/12.0)
    X_train['month_cos'] = -np.cos(2 * np.pi * X_train['month']/12.0)
    ## cyclic transformation on weekday
    X_train['weekday_sin'] = -np.sin(2 * np.pi * (X_train['weekday']+1)/7.0)
    X_train['weekday_cos'] = -np.cos(2 * np.pi * (X_train['weekday']+1)/7.0)
    ## min temperature
    X_train = X_train.merge(X_train.groupby(['num','date'])['temperature'].min().reset_index().rename(columns = {'temperature':'min_temperature'}), on = ['num','date'], how = 'left')
    ## THI
    X_train['THI'] = 9/5*X_train['temperature'] - 0.55*(1-X_train['humidity']/100)*(9/5*X_train['temperature']-26)+32
    ## mean THI
    X_train = X_train.merge(X_train.groupby(['num','date'])['THI'].mean().reset_index().rename(columns = {'THI':'mean_THI'}), on = ['num','date'], how = 'left')
    ## CDH 
    cdhs = np.array([])
    for num in range(1,61,1):
        temp = X_train[X_train['num'] == num]
        cdh = CDH(temp['temperature'].values)
        cdhs = np.concatenate([cdhs, cdh])
    X_train['CDH'] = cdhs
    ## mean CDH
    X_train = X_train.merge(X_train.groupby(['num','date'])['CDH'].mean().reset_index().rename(columns = {'CDH':'mean_CDH'}), on = ['num','date'], how = 'left')
    # droping unnecessry columns
    X_train.drop(['solar_flag', 'nelec_cool_flag'], axis=1, inplace=True)

    # split dataframe to separately modeling for each buildings 
    X_trains = [X_train[X_train.num == num] for num in range(1,61,1)]

    # applying seperate feature engineering for each clusters
    for num in [3,9,12,21,24,34,51]:
            temp_df = X_trains[num-1]
            daily_insol = temp_df.groupby(['date'])['insolation'].sum().reset_index()
            daily_insol_1b = daily_insol['insolation'].shift(1)
            daily_insol['insolation_1c'] = (daily_insol['insolation'] + daily_insol_1b).fillna(method = 'bfill')
            daily_insol.drop(['insolation'], axis = 1, inplace = True)
            temp_df = temp_df.merge(daily_insol, on = 'date', how = 'left')
            X_trains[num-1] = temp_df
    ## cluster 0
    for num in clust_to_num[0]:
        temp_df = X_trains[num-1]
        temp_df['aug_day'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['month']==8)).astype(int)
        X_trains[num-1] = temp_df
    for num in [4,30,36,10]:
        temp_df = X_trains[num-1]
        temp_df['19-07'] = ((temp_df['hour']>=19)|(temp_df['hour']<=7)).astype(int)
        temp_df['08-18'] = 1-temp_df['19-07']
        X_trains[num-1] = temp_df
    for num in [11,12,41,40,42,28]:
        temp_df = X_trains[num-1]
        temp_df['21-09'] = ((temp_df['hour']>=21)|(temp_df['hour']<=9)).astype(int)
        temp_df['10-20'] = 1-temp_df['21-09']
        X_trains[num-1] = temp_df
    for num in [29,60]:
        temp_df = X_trains[num-1]
        temp_df['22-06'] = ((temp_df['hour']>=22)|(temp_df['hour']<=6)).astype(int)
        temp_df['07-21'] = 1-temp_df['22-06']
        X_trains[num-1] = temp_df
    for num in [10]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[~temp_df['date'].map(str).isin(['2020-07-27','2020-08-10'])]
        X_trains[num-1] = temp_df
    for num in [40]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[~temp_df['date'].map(str).isin(['2020-08-03'])]
        X_trains[num-1] = temp_df
    for num in [42]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[~temp_df['date'].map(str).isin(['2020-08-10'])]
        X_trains[num-1] = temp_df
    ## cluster 1
    for num in [1]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[temp_df['date'] > pd.to_datetime('2020-06-05')]
        temp_df['weekend'] = (temp_df['weekday'].isin([5,6])).astype(int)
        temp_df['c_02-08'] = ((temp_df['weekday'].isin([0,1,2,3,4]))&(temp_df['hour']>=2)&(temp_df['hour']<=8)).astype(int)
        temp_df['e_05-09'] = ((temp_df['weekday'].isin([5,6]))&(temp_df['hour']>=5)&(temp_df['hour']<=9)).astype(int)
        temp_df['m_10-18'] = ((temp_df['weekday'].isin([0]))&(temp_df['hour']>=10)&(temp_df['hour']<=18)).astype(int)
        X_trains[num-1] = temp_df
    for num in [5]:
        temp_df = X_trains[num-1]
        temp_df['fs_19-04'] = ((temp_df['weekday'].isin([4]))&(temp_df['hour']>=19))|((temp_df['weekday'].isin([5]))&(temp_df['hour']>=4))
        temp_df['05-08'] = ((temp_df['hour']>=5)&(temp_df['hour']<=8)).astype(int)
        temp_df['18'] = (temp_df['hour']==18).astype(int)
        X_trains[num-1] = temp_df
    for num in [9]:
        temp_df = X_trains[num-1]
        temp_df['19-05'] = ((temp_df['hour']>=19)|(temp_df['hour']<=5)).astype(int)
        temp_df['ts_06-16'] = (((temp_df['weekday'].isin([1]))&(temp_df['hour']>=6)&(temp_df['hour']<=16))|((temp_df['weekday'].isin([6]))&(temp_df['hour']>=6)&(temp_df['hour']<=16))).astype(int)
        temp_df = temp_df[~temp_df['date'].map(str).isin(['2020-08-17','2020-08-16'])]
        X_trains[num-1] = temp_df
    for num in [34]:
        temp_df = X_trains[num-1]
        temp_df['t-s_18-21'] = ((temp_df['weekday'].isin([1,2,3,4,5,6]))&(temp_df['hour']>=18)&(temp_df['hour']<=21)).astype(int)
        X_trains[num-1] = temp_df
    ## cluster 2
    for num in clust_to_num[2]:
        temp_df = X_trains[num-1]
        temp_df['aug_night'] = ((temp_df['hour'].isin([18,19,20,21,22]))&(temp_df['month']==8)).astype(int)
        X_trains[num-1] = temp_df
    for num in [19,20,21,49,50]:
        temp_df = X_trains[num-1]
        temp_df['weekend'] = (temp_df['weekday'].isin([5,6])).astype(int)
        temp_df['01-06'] = ((temp_df['hour']>=1)&(temp_df['hour']<=6)).astype(int)
        temp_df['18-22'] = ((temp_df['hour']>=18)&(temp_df['hour']<=22)).astype(int)
        X_trains[num-1] = temp_df
    ## cluster 3
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[~temp_df['date'].map(str).isin(['2020-08-17'])]
        X_trains[num-1] = temp_df
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df['aug_day'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['month']==8)&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df['weekend'] = ((temp_df['weekday'].isin([5,6]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [2,6,13,14,16,22,23,26,27,35,37,44,52,53,3,8,31,33,24]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['weekday'].isin([0,1,2,3,4]))&(temp_df['hour'] >= 8)&(temp_df['hour'] <= 18)).astype(int)
        X_trains[num-1] = temp_df
    for num in [3,25,26,48,54,55,56]:
        temp_df = X_trains[num-1]
        temp_df['lunch_time'] = ((temp_df['hour'].isin([11,12]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [3]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[(temp_df['date']<pd.to_datetime('2020-07-14'))|(temp_df['date']>=pd.to_datetime('2020-08-10'))]
        X_trains[num-1] = temp_df
    for num in [7]:
        temp_df = X_trains[num-1]
        temp_df['23-02'] = (temp_df['hour'].isin([23,0,1,2])).astype(int)
        temp_df['working_time'] = ((temp_df['weekday'].isin([0,1,2,3,4]))&(temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [15]:
        temp_df = X_trains[num-1]
        temp_df['c_08'] = ((temp_df['hour'].isin([8]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['e_09-22'] = ((temp_df['hour'].isin([9,10,11,12,13,14,15,16,17,18,19,20,21,22]))&(temp_df['weekday'].isin([5,6]))).astype(int)
        temp_df['19-22'] = (temp_df['hour'].isin([19,20,21,22])).astype(int)
        X_trains[num-1] = temp_df
    for num in [18]:
        temp_df = X_trains[num-1]
        temp_df['00-03'] = (temp_df['hour'].isin([0,1,2,3])).astype(int)
        temp_df['working_time'] = ((temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['e_15-17'] = ((temp_df['hour'].isin([15,16,17]))&(temp_df['weekday'].isin([6]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [23]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[(temp_df['date']>=pd.to_datetime('2020-08-13'))]
        X_trains[num-1] = temp_df
    for num in [24]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[(temp_df['date']>=pd.to_datetime('2020-06-04'))]
        temp_df['s_0-6'] = ((temp_df['weekday']==6)&(temp_df['hour'].isin([0,1,2,3,4,5,6]))).astype(int)
        temp_df['09-18'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17,18])).astype(int)
        temp_df['06-21'] = (temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])).astype(int)
        X_trains[num-1] = temp_df
    for num in [25]:
        temp_df = X_trains[num-1]
        temp_df['working_time_9-16'] = ((temp_df['hour'].isin([9,10,11,12,13,14,15,16]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['working_time_8-18'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df = temp_df[(temp_df['date']<pd.to_datetime('2020-07-26'))|(temp_df['date']>=pd.to_datetime('2020-08-03'))]
        X_trains[num-1] = temp_df
    for num in [27]:
        temp_df = X_trains[num-1]
        temp_df = temp_df[(temp_df['datetime']<pd.to_datetime('2020-08-08 12:00:00'))|(temp_df['datetime']>pd.to_datetime('2020-08-08 16:00:00'))]
        X_trains[num-1] = temp_df
    for num in [38]:
        temp_df = X_trains[num-1]
        temp_df['22-04'] = (temp_df['hour'].isin([0,1,2,3,4,22,23]))
        temp_df['working_time'] = ((temp_df['hour'].isin([7,8,9,10,11,12,13,14,15,16,17,18]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [39]:
        temp_df = X_trains[num-1]
        temp_df['09-19'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17,18,19])).astype(int)
        X_trains[num-1] = temp_df
    for num in [43]:
        temp_df = X_trains[num-1]
        temp_df['00-06'] = (temp_df['hour'].isin([0,1,2,3,4,5,6])).astype(int)
        X_trains[num-1] = temp_df
    for num in [44]:
        temp_df = X_trains[num-1]
        temp_df['08-17'] = (temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17])).astype(int)
        X_trains[num-1] = temp_df
    for num in [45]:
        temp_df = X_trains[num-1]
        temp_df['07-12'] = (temp_df['hour'].isin([7,8,9,10,11,12])).astype(int)
        temp_df['13-17'] = (temp_df['hour'].isin([13,14,15,16,17])).astype(int)
        X_trains[num-1] = temp_df
    for num in [46]:
        temp_df = X_trains[num-1]
        temp_df['06-18'] = (temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18])).astype(int)
        X_trains[num-1] = temp_df
    for num in [47]:
        temp_df = X_trains[num-1]
        temp_df['08-17'] = (temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17])).astype(int)
        temp_df['18-20'] = (temp_df['hour'].isin([18,19,20])).astype(int)
        X_trains[num-1] = temp_df
    for num in [48]:
        temp_df = X_trains[num-1]
        temp_df['dark'] = (temp_df['hour'].isin([8,18,19,20])).astype(int)
        temp_df['09-17'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17])).astype(int)
        X_trains[num-1] = temp_df
    for num in [54]:
        temp_df = X_trains[num-1]
        temp_df['dark'] = (temp_df['hour'].isin([5,6,7,17,18,19])).astype(int)
        temp_df['08-16'] = (temp_df['hour'].isin([8,9,10,11,12,13,14,15,16])).astype(int)
        X_trains[num-1] = temp_df
    for num in [55,56]:
        temp_df = X_trains[num-1]
        temp_df['09-17'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17])).astype(int)
        temp_df['18-21'] = (temp_df['hour'].isin([18,19,20,21])).astype(int)
        temp_df = temp_df[(temp_df['date']<pd.to_datetime('2020-08-03'))|(temp_df['date']>=pd.to_datetime('2020-08-10'))]
        X_trains[num-1] = temp_df
    for num in [57]:
        temp_df = X_trains[num-1]
        temp_df['m-s00-05'] = ((temp_df['hour'].isin([0,1,2,3,4,5]))&(temp_df['weekday'].isin([0,1,2,3,4,5]))).astype(int)
        temp_df['s_dark'] = ((temp_df['hour'].isin([10,11,13,14,15]))&(temp_df['weekday'].isin([5]))).astype(int)
        temp_df['06-07'] = (temp_df['hour'].isin([6,7])).astype(int)
        X_trains[num-1] = temp_df
    for num in [58]:
        temp_df = X_trains[num-1]
        temp_df['c_08-17'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['22-04'] = (temp_df['hour'].isin([22,23,0,1,2,3,4])).astype(int)
        X_trains[num-1] = temp_df
        
    # weather relevant feature for each building
    for num in range(1,61,1):
        temp_df = X_trains[num-1]
        temp_df['THI_cat'] = pd.cut(temp_df['THI'], bins = [0, 68, 75, 80, 200], labels = [1,2,3,4])
        X_trains[num-1] = temp_df

    y_trains = [df['target'].values for df in X_trains]
    X_trains_ohe = [df.drop('target', axis = 1) for df in X_trains]
    y_trains_log = [np.log(df) for df in y_trains]

    ## one hot encoding for weekday, hour, THI
    for i, X_train in enumerate(X_trains_ohe):
        X_train = pd.concat([X_train, pd.get_dummies(X_train['weekday'], prefix ='weekday')], axis=1)
        X_train = pd.concat([X_train, pd.get_dummies(X_train['hour'], prefix ='hour')], axis=1)
        X_train = pd.concat([X_train, pd.get_dummies(X_train['THI_cat'], prefix ='THI')], axis=1)
        X_trains_ohe[i] = X_train

    # drop unnecessary columns
    X_trains_ohe = [df.drop(['num', 'datetime', 'day', 'date', 'weekday', 'hour', 'month', 'THI_cat'], axis=1).reset_index().drop('index', axis=1) for df in X_trains_ohe]

    # standard scaling for num features on X_trains_ohe
    num_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'hour_sin', 'hour_cos', 'date_sin', 'date_cos','month_sin','month_cos','weekday_sin','weekday_cos','min_temperature','THI','mean_THI','CDH','mean_CDH']
    num_features_solar = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'hour_sin', 'hour_cos', 'date_sin', 'date_cos','month_sin','month_cos','weekday_sin','weekday_cos','min_temperature','THI','mean_THI','CDH','mean_CDH','insolation_1c']
    means = []
    stds = []
    for i, df in enumerate(X_trains_ohe):
        if i+1 in [3,9,12,21,24,34,51]:
            means.append(df.loc[:,num_features_solar].mean(axis=0))
            stds.append(df.loc[:,num_features_solar].std(axis=0))
            df.loc[:,num_features_solar] = (df.loc[:,num_features_solar] - df.loc[:,num_features_solar].mean(axis=0))/df.loc[:,num_features_solar].std(axis=0)
        else:
            means.append(df.loc[:,num_features].mean(axis=0))
            stds.append(df.loc[:,num_features].std(axis=0))
            df.loc[:,num_features] = (df.loc[:,num_features] - df.loc[:,num_features].mean(axis=0))/df.loc[:,num_features].std(axis=0)
        X_trains_ohe[i] = df

    return X_trains_ohe, y_trains_log, means, stds


# data preprocessing function for testset
def test_preprocess(test, clust_to_num, means, stds):
    '''
    train : test dataframe
    clust_to_num : dict containing cluster as a key, according buildings as values
    means : list containing np.mean(train, axis = 0) of every 60 buildings
    stds : list containing np.std(train, axis = 0) of every 60 buildings
    '''
    X_train = test.copy()
    X_train = X_train.interpolate()
    
    X_train['datetime'] = pd.to_datetime(X_train['datetime'])
    X_train['hour'] = X_train['datetime'].dt.hour
    X_train['month'] = X_train['datetime'].dt.month
    X_train['day'] = X_train['datetime'].dt.day
    X_train['date'] = X_train['datetime'].dt.date
    X_train['weekday'] = X_train['datetime'].dt.weekday
    
    # feature engineering universally applied to every building

    ## cyclic transformation on hour
    X_train['hour_sin'] = np.sin(2 * np.pi * X_train['hour']/23.0)
    X_train['hour_cos'] = np.cos(2 * np.pi * X_train['hour']/23.0)
    ## cyclic transformation on date 
    X_train['date_sin'] = -np.sin(2 * np.pi * (X_train['month']+X_train['day']/31)/12.0)
    X_train['date_cos'] = -np.cos(2 * np.pi * (X_train['month']+X_train['day']/31)/12.0)
    ## cyclic transformation on month
    X_train['month_sin'] = -np.sin(2 * np.pi * X_train['month']/12.0)
    X_train['month_cos'] = -np.cos(2 * np.pi * X_train['month']/12.0)
    ## cyclic transformation on weekday
    X_train['weekday_sin'] = -np.sin(2 * np.pi * (X_train['weekday']+1)/7.0)
    X_train['weekday_cos'] = -np.cos(2 * np.pi * (X_train['weekday']+1)/7.0)
    ## daily minimum temperature
    X_train = X_train.merge(X_train.groupby(['num','date'])['temperature'].min().reset_index().rename(columns = {'temperature':'min_temperature'}), on = ['num','date'], how = 'left')
    ## THI
    X_train['THI'] = 9/5*X_train['temperature'] - 0.55*(1-X_train['humidity']/100)*(9/5*X_train['temperature']-26)+32
    ## mean_THI
    X_train = X_train.merge(X_train.groupby(['num','date'])['THI'].mean().reset_index().rename(columns = {'THI':'mean_THI'}), on = ['num','date'], how = 'left')
    ## CDH
    cdhs = np.array([])
    for num in range(1,61,1):
        temp = X_train[X_train['num'] == num]
        cdh = CDH(temp['temperature'].values)
        cdhs = np.concatenate([cdhs, cdh])
    X_train['CDH'] = cdhs
    ## mean_CDH
    X_train = X_train.merge(X_train.groupby(['num','date'])['CDH'].mean().reset_index().rename(columns = {'CDH':'mean_CDH'}), on = ['num','date'], how = 'left')
    # droping unnecessry columns
    X_train.drop(['solar_flag', 'nelec_cool_flag'], axis=1, inplace=True)
    
    X_trains = [X_train[X_train.num == num] for num in range(1,61,1)]
    
    # applying seperate feature engineering for each clusters
    ## solar flag == 1
    for num in [3,9,12,21,24,34,51]:
        temp_df = X_trains[num-1]
        daily_insol = temp_df.groupby(['date'])['insolation'].sum().reset_index()
        daily_insol_1b = daily_insol['insolation'].shift(1)
        daily_insol['insolation_1c'] = (daily_insol['insolation'] + daily_insol_1b).fillna(method = 'bfill')
        daily_insol.drop(['insolation'], axis = 1, inplace = True)
        temp_df = temp_df.merge(daily_insol, on = 'date', how = 'left')
        X_trains[num-1] = temp_df
    ## cluster 0
    for num in clust_to_num[0]:
        temp_df = X_trains[num-1]
        temp_df['aug_day'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['month']==8)).astype(int)
        X_trains[num-1] = temp_df
    for num in [4,30,36,10]:
        temp_df = X_trains[num-1]
        temp_df['19-07'] = ((temp_df['hour']>=19)|(temp_df['hour']<=7)).astype(int)
        temp_df['08-18'] = 1-temp_df['19-07']
        X_trains[num-1] = temp_df
    for num in [11,12,41,40,42,28]:
        temp_df = X_trains[num-1]
        temp_df['21-09'] = ((temp_df['hour']>=21)|(temp_df['hour']<=9)).astype(int)
        temp_df['10-20'] = 1-temp_df['21-09']
        X_trains[num-1] = temp_df
    for num in [29,60]:
        temp_df = X_trains[num-1]
        temp_df['22-06'] = ((temp_df['hour']>=22)|(temp_df['hour']<=6)).astype(int)
        temp_df['07-21'] = 1-temp_df['22-06']
        X_trains[num-1] = temp_df
    ## cluster 1
    for num in [1]:
        temp_df = X_trains[num-1]
        temp_df['weekend'] = (temp_df['weekday'].isin([5,6])).astype(int)
        temp_df['c_02-08'] = ((temp_df['weekday'].isin([0,1,2,3,4]))&(temp_df['hour']>=2)&(temp_df['hour']<=8)).astype(int)
        temp_df['e_05-09'] = ((temp_df['weekday'].isin([5,6]))&(temp_df['hour']>=5)&(temp_df['hour']<=9)).astype(int)
        temp_df['m_10-18'] = ((temp_df['weekday'].isin([0]))&(temp_df['hour']>=10)&(temp_df['hour']<=18)).astype(int)
        X_trains[num-1] = temp_df
    for num in [5]:
        temp_df = X_trains[num-1]
        temp_df['fs_19-04'] = ((temp_df['weekday'].isin([4]))&(temp_df['hour']>=19))|((temp_df['weekday'].isin([5]))&(temp_df['hour']>=4))
        temp_df['05-08'] = ((temp_df['hour']>=5)&(temp_df['hour']<=8)).astype(int)
        temp_df['18'] = (temp_df['hour']==18).astype(int)
        X_trains[num-1] = temp_df
    for num in [9]:
        temp_df = X_trains[num-1]
        temp_df['19-05'] = ((temp_df['hour']>=19)|(temp_df['hour']<=5)).astype(int)
        temp_df['ts_06-16'] = (((temp_df['weekday'].isin([1]))&(temp_df['hour']>=6)&(temp_df['hour']<=16))|((temp_df['weekday'].isin([6]))&(temp_df['hour']>=6)&(temp_df['hour']<=16))).astype(int)
        X_trains[num-1] = temp_df
    for num in [34]:
        temp_df = X_trains[num-1]
        temp_df['t-s_18-21'] = ((temp_df['weekday'].isin([1,2,3,4,5,6]))&(temp_df['hour']>=18)&(temp_df['hour']<=21)).astype(int)
        X_trains[num-1] = temp_df
    ## cluster 2
    for num in clust_to_num[2]:
        temp_df = X_trains[num-1]
        temp_df['aug_night'] = ((temp_df['hour'].isin([18,19,20,21,22]))&(temp_df['month']==8)).astype(int)
        X_trains[num-1] = temp_df
    for num in [19,20,21,49,50]:
        temp_df = X_trains[num-1]
        temp_df['weekend'] = (temp_df['weekday'].isin([5,6])).astype(int)
        temp_df['01-06'] = ((temp_df['hour']>=1)&(temp_df['hour']<=6)).astype(int)
        temp_df['18-22'] = ((temp_df['hour']>=18)&(temp_df['hour']<=22)).astype(int)
        X_trains[num-1] = temp_df
    ## cluster 3
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df['aug_day'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['month']==8)&temp_df['weekday'].isin([0,1,2,3,4])).astype(int)
        X_trains[num-1] = temp_df
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df['weekend'] = ((temp_df['weekday'].isin([5,6]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [2,6,13,14,16,22,23,26,27,35,37,44,52,53,3,8,31,33,24]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['weekday'].isin([0,1,2,3,4]))&(temp_df['hour'] >= 8)&(temp_df['hour'] <= 18)).astype(int)
        X_trains[num-1] = temp_df
    for num in [3,25,26,48,54,55,56]:
        temp_df = X_trains[num-1]
        temp_df['lunch_time'] = ((temp_df['hour'].isin([11,12]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [7]:
        temp_df = X_trains[num-1]
        temp_df['23-02'] = (temp_df['hour'].isin([23,0,1,2])).astype(int)
        temp_df['working_time'] = ((temp_df['weekday'].isin([0,1,2,3,4]))&(temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [15]:
        temp_df = X_trains[num-1]
        temp_df['c_08'] = ((temp_df['hour'].isin([8]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['e_09-22'] = ((temp_df['hour'].isin([9,10,11,12,13,14,15,16,17,18,19,20,21,22]))&(temp_df['weekday'].isin([5,6]))).astype(int)
        temp_df['19-22'] = (temp_df['hour'].isin([19,20,21,22])).astype(int)
        X_trains[num-1] = temp_df
    for num in [18]:
        temp_df = X_trains[num-1]
        temp_df['00-03'] = (temp_df['hour'].isin([0,1,2,3])).astype(int)
        temp_df['working_time'] = ((temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['e_15-17'] = ((temp_df['hour'].isin([15,16,17]))&(temp_df['weekday'].isin([6]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [24]:
        temp_df = X_trains[num-1]
        temp_df['s_0-6'] = ((temp_df['weekday']==6)&(temp_df['hour'].isin([0,1,2,3,4,5,6]))).astype(int)
        temp_df['09-18'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17,18])).astype(int)
        temp_df['06-21'] = (temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])).astype(int)
        X_trains[num-1] = temp_df
    for num in [25]:
        temp_df = X_trains[num-1]
        temp_df['working_time_9-16'] = ((temp_df['hour'].isin([9,10,11,12,13,14,15,16]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['working_time_8-18'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df = temp_df[(temp_df['date']<pd.to_datetime('2020-07-26'))|(temp_df['date']>=pd.to_datetime('2020-08-03'))]
        X_trains[num-1] = temp_df
    for num in [38]:
        temp_df = X_trains[num-1]
        temp_df['22-04'] = (temp_df['hour'].isin([0,1,2,3,4,22,23]))
        temp_df['working_time'] = ((temp_df['hour'].isin([7,8,9,10,11,12,13,14,15,16,17,18]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    for num in [39]:
        temp_df = X_trains[num-1]
        temp_df['09-19'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17,18,19])).astype(int)
        X_trains[num-1] = temp_df
    for num in [43]:
        temp_df = X_trains[num-1]
        temp_df['00-06'] = (temp_df['hour'].isin([0,1,2,3,4,5,6])).astype(int)
        X_trains[num-1] = temp_df
    for num in [44]:
        temp_df = X_trains[num-1]
        temp_df['08-17'] = (temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17])).astype(int)
        X_trains[num-1] = temp_df
    for num in [45]:
        temp_df = X_trains[num-1]
        temp_df['07-12'] = (temp_df['hour'].isin([7,8,9,10,11,12])).astype(int)
        temp_df['13-17'] = (temp_df['hour'].isin([13,14,15,16,17])).astype(int)
        X_trains[num-1] = temp_df
    for num in [46]:
        temp_df = X_trains[num-1]
        temp_df['06-18'] = (temp_df['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18])).astype(int)
        X_trains[num-1] = temp_df
    for num in [47]:
        temp_df = X_trains[num-1]
        temp_df['08-17'] = (temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17])).astype(int)
        temp_df['18-20'] = (temp_df['hour'].isin([18,19,20])).astype(int)
        X_trains[num-1] = temp_df
    for num in [48]:
        temp_df = X_trains[num-1]
        temp_df['dark'] = (temp_df['hour'].isin([8,18,19,20])).astype(int)
        temp_df['09-17'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17])).astype(int)
        X_trains[num-1] = temp_df
    for num in [54]:
        temp_df = X_trains[num-1]
        temp_df['dark'] = (temp_df['hour'].isin([5,6,7,17,18,19])).astype(int)
        temp_df['08-16'] = (temp_df['hour'].isin([8,9,10,11,12,13,14,15,16])).astype(int)
        X_trains[num-1] = temp_df
    for num in [55,56]:
        temp_df = X_trains[num-1]
        temp_df['09-17'] = (temp_df['hour'].isin([9,10,11,12,13,14,15,16,17])).astype(int)
        temp_df['18-21'] = (temp_df['hour'].isin([18,19,20,21])).astype(int)
        X_trains[num-1] = temp_df
    for num in [57]:
        temp_df = X_trains[num-1]
        temp_df['m-s00-05'] = ((temp_df['hour'].isin([0,1,2,3,4,5]))&(temp_df['weekday'].isin([0,1,2,3,4,5]))).astype(int)
        temp_df['s_dark'] = ((temp_df['hour'].isin([10,11,13,14,15]))&(temp_df['weekday'].isin([5]))).astype(int)
        temp_df['06-07'] = (temp_df['hour'].isin([6,7])).astype(int)
        X_trains[num-1] = temp_df
    for num in [58]:
        temp_df = X_trains[num-1]
        temp_df['c_08-17'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['22-04'] = (temp_df['hour'].isin([22,23,0,1,2,3,4])).astype(int)
        X_trains[num-1] = temp_df
    # weather relevant feature for each building
    for num in range(1,61,1):
        temp_df = X_trains[num-1]
        temp_df['THI_cat'] = pd.cut(temp_df['THI'], bins = [0, 68, 75, 80, 200], labels = [1,2,3,4])
        X_trains[num-1] = temp_df
    
    X_trains_ohe = X_trains.copy()
    for i, X_train in enumerate(X_trains_ohe):
        X_train = pd.concat([X_train, pd.get_dummies(X_train['weekday'], prefix ='weekday')], axis=1)
        X_train = pd.concat([X_train, pd.get_dummies(X_train['hour'], prefix ='hour')], axis=1)
        X_train = pd.concat([X_train, pd.get_dummies(X_train['THI_cat'], prefix ='THI')], axis=1)
        X_trains_ohe[i] = X_train
    # drop unnecessary columns
    X_trains_ohe = [df.drop(['num', 'datetime', 'hour', 'month', 'day', 'date', 'weekday', 'THI_cat'], axis=1).reset_index().drop('index', axis=1) for df in X_trains_ohe]
    # standard scaling for numerical features on X_tests and X_tests_ohe
    num_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'hour_sin', 'hour_cos', 'date_sin', 'date_cos','month_sin','month_cos','weekday_sin','weekday_cos','min_temperature','THI', 'mean_THI','CDH','mean_CDH']
    num_features_solar = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation', 'hour_sin', 'hour_cos', 'date_sin', 'date_cos','month_sin','month_cos','weekday_sin','weekday_cos','min_temperature','THI', 'mean_THI','CDH','mean_CDH','insolation_1c']
    
    for i, (df, mean, std) in enumerate(zip(X_trains_ohe, means, stds)):
        if i+1 in [3,9,12,21,24,34,51]:
            df.loc[:,num_features_solar] = (df.loc[:,num_features_solar] - mean) / std
        else:
            df.loc[:,num_features] = (df.loc[:,num_features] - mean) / std
        X_trains_ohe[i] = df
   
    return X_trains, X_trains_ohe
