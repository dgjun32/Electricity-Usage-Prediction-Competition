import os
import sys
import numpy as np
import pandas as pd

# Learning algorithms
import sklearn
from sklearn.linear_model import *
from sklearn.svm import SVR
from sklearn.cluster import KMeans
import lightgbm as lgb
from lightgbm import LGBMRegressor
import catboost
from catboost import CatBoostRegressor

# model validation
from sklearn.model_selection import KFold
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

SEED = 2
np.random.seed(SEED)

from config import cfg
from preprocess import preprocess_df, test_preprocess
from ensemble import good_models

if __name__ == 'main':
    # 1. load dataset
    train_df = pd.read_csv(cfg.dir.train_df)
    test_df = pd.read_csv(cfg.dir.test_df)
    sub_df = pd.read_csv(cfg.dir.submission_df)

    # 2. renaming columns of dataframe
    train_df.columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
    test_df.columns = ['num','datetime','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']

    # 3. K-Means clustering on buildings for preprocessing
    by_weekday = train_df.groupby(['num','weekday'])['target'].median().reset_index().pivot('num','weekday','target').reset_index()
    by_hour = train_df.groupby(['num','hour'])['target'].median().reset_index().pivot('num','hour','target').reset_index().drop('num', axis = 1)
    df = pd.concat([by_weekday, by_hour], axis= 1)
    columns = ['num'] + ['day'+str(i) for i in range(7)] + ['hour'+str(i) for i in range(24)]
    df.columns = columns

    for i in range(len(df)):
        df.iloc[i,1:8] = (df.iloc[i,1:8] - df.iloc[i,1:8].mean())/df.iloc[i,1:8].std()
        df.iloc[i,8:] = (df.iloc[i,8:] - df.iloc[i,8:].mean())/df.iloc[i,8:].std()

    kmeans = KMeans(n_clusters=4, random_state = 2)
    km_cluster = kmeans.fit_predict(df.iloc[:,1:])

    df_clust = df.copy()
    df_clust['km_cluster'] = km_cluster
    df_clust['km_cluster'] = df_clust['km_cluster'].map({0:1, 1:3, 2:2, 3:0})
    
    match = df_clust[['num','km_cluster']]

    # dictionary mapping cluster and building number
    clust_to_num = {0:[],1:[],2:[],3:[]}
    for i in range(60):
        c = match.iloc[i,1]
        clust_to_num[c].append(i+1)

    # 4. Preprocessing data
    X_trains_ohe, y_trains_log, means, stds = preprocess_df(train_df, clust_to_num)
    X_tests_ohe = test_preprocess(test_df, clust_to_num, means, stds)

    # 5. Model Ensemble Strategy : Pick good models that perform well on each buildings
    cat_best_scores = cfg.cv_scores.cat_best_scores
    lgb_best_scores = cfg.cv_scores.lgb_best_scores
    svr_best_scores = cfg.cv_scores.svr_best_scores
    ll_best_scores = cfg.cv_scores.ll_best_scores
    enet_best_scores = cfg.cv_scores.enet_best_scores
    score_df = pd.DataFrame({'model':['cat']*60 + ['lgb']*60 + ['enet']*60 + ['lassolars']*60+['svr']*60,
                         'building': list(range(1,61))*5,
                         'smape' : cat_best_scores + lgb_best_scores + enet_best_scores + ll_best_scores + svr_best_scores})
    score_df['smape'] = score_df['smape'].abs()

    # pick model that perform superior prediction on certain building
    best_models = good_models(score_df, 0.3, 1.1)

    cat_hyperparams = cfg.hyperparams.cat_hyperparams
    lgb_hyperparams = cfg.hyperparams.lgb_hyperparams
    enet_hyperparams = cfg.hyperparams.enet_hyperparams
    ll_hyperparams = cfg.hyperparams.ll_hyperparams
    svr_hyperparams = cfg.hyperparams.svr_hyperparams

    # catboost models with tuned hyperparams
    opt_cat = [CatBoostRegressor(random_state = SEED, verbose = False, **params) for params in cat_hyperparams]
    # lgbm models with tuned hyperparams
    opt_lgb = [LGBMRegressor(random_state = SEED, verbose = 0, **params) for params in lgb_hyperparams]
    # elasticnet models with tuned hyperparams
    opt_enet = [ElasticNet(random_state = SEED, **params) for params in enet_hyperparams]
    # lassolars with tuned hyperparams
    opt_ll = [LassoLars(**params) for params in ll_hyperparams]
    # SVR regressor with tuned hyperparams
    opt_svr = [SVR(**params) for params in svr_hyperparams]

    # 6. voting ensemble training & save & inference
    bests = good_models(score_df, 0.3, 1.1)
    voting_pred = []
    for i, (X_test, X_tr, y_tr, best, c, l, e, ll, s) in enumerate(zip(X_tests_ohe, X_trains_ohe, y_trains_log, bests, opt_cat, opt_lgb, opt_enet, opt_ll, opt_svr)):
        os.mkdir('../model/building_{}/'.format(i+1))
        pred = []
        if 'cat' in best:
            cat = c
            cat.fit(X_tr, y_tr, verbose = False)
            pred.append(cat.predict(X_test))
            joblib.dump(cat, '../model/building_{}/catboost.pkl'.format(i+1))
        if 'lgb' in best:
            lgb = l
            lgb.fit(X_tr, y_tr, verbose = 0)
            pred.append(lgb.predict(X_test))
            joblib.dump(lgb, '../model/building_{}/lgbm.pkl'.format(i+1))
        if 'enet' in best:
            enet = e
            enet.fit(X_tr, y_tr)
            pred.append(enet.predict(X_test))
            joblib.dump(enet, '../model/building_{}/elasticnet.pkl'.format(i+1))
        if 'lassolars' in best:
            lassolars = ll
            lassolars.fit(X_tr, y_tr)
            pred.append(lassolars.predict(X_test))
            joblib.dump(cat, '../model/building_{}/lassolars.pkl'.format(i+1))
        if 'svr' in best:
            svr = s
            svr.fit(X_tr, y_tr)
            pred.append(svr.predict(X_test))
            joblib.dump(svr, '../model/building_{}/svr.pkl'.format(i+1))
        voting_pred.append(np.exp(np.array(pred).mean(axis = 0)))

        print (f'model{i+1} training complete')
    voting_pred = np.concatenate(voting_pred)
    sub_df['answer'] = voting_pred
    sub_df.to_csv('submission.csv')


    