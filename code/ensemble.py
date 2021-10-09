import pandas as pd
import numpy as np

def good_models(score_df, pivot_q, threshold):
    score_pivot = pd.DataFrame(score_df.pivot('building', 'model', 'smape').values,
                               columns = ['cat','enet','lassolars','lgb', 'svr'])
    li = []
    for i in range(len(score_pivot)):
        temp = score_pivot.iloc[i]
        q = temp.quantile(pivot_q)
        best = list(temp[temp <= threshold*q].index)
        li.append(best)
    return li

