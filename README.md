# Building-wise Regression Ensemble for Electricity Usage Prediction Competition
#### <b>Top 2% in Private Leaderboard</b> and <b>Silver prize in EDA notebook competition field.</b>

## 1. Methodology
### 1.1. Summary of EDA
According to Exploratory Data Analysis, which you can check in my notebook file in this repository, Each 60 buildings have distinguishing behavior of electricity consumption pattern. 

<img src = '../figures/heatmap.png'>

Therefore, I determined to <b>apply seperate feature engineering</b> and <b>train seperate model</b> for each  60 buildings. 

Also, I conducted K-means clustering on electricity usage pattern of each buildings. I assigned cluster from 0 to 3, which buildings in the same cluster share similar electricity usage pattern. I applied same feature engineering to buildings within same cluster. Below is the result of clustering.

<img src = '../figures/heatmap_cluster.png'>

### 1.2. Model Ensembling Strategy

I initially computed 8 fold CV score of five models ```CatBoostRegressor``` ```LGBMRegressor``` ```SVR```
```ElasticNet``` ```LassoLars``` for each 60 buildings. (CV score of 5 models for each 60 buildings, as a total, 300 CV scores).

However, according to CV scores, some models perform well on certain buildings but not on other buildings. Below is the visualization of CV scores.

<img src = '../figures/cv_scores.png'>

Therefore, I implemented function for selecting 'good models for each building' based on CV score.

```
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
```

By using ```good_models``` function with parameter ```pivot_q = 0.3``` and ```threshold = 1.1```, I selected models for ensemble in each building.

## 2. Data
Download .csv files in this link and save it as ```train.csv```, ```test.csv```, ```submission.csv``` at ```../data/```.

## 3. Training
```python main.py```

This will train models and save trained models in ```../model/```.

## 4. Evaluation
