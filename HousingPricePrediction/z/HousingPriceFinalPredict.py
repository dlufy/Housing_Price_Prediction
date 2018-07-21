
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
import time
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, LassoCV,LassoLarsCV, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error as mae


# In[57]:


def DataPreprocess(train,test):
    #concating train and test data to make suitable for fiiting model and predicting
    ConcatedData = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
    ConcatedData.head()
    #these values are have a very few detial available we should drop these columns
    toDel = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
    ConcatedData = ConcatedData.drop(toDel,axis=1)
    ##Taking log of values to make them scaleable
    train["SalePrice"] = np.log1p(train["SalePrice"])
    #finding numeric only columns
    numeric_feats = ConcatedData.dtypes[ConcatedData.dtypes != "object"].index
    #computing skewness
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    #taking log to make data scaleable
    ConcatedData[skewed_feats] = np.log1p(ConcatedData[skewed_feats])
    #one hot key encoding
    ConcatedData = pd.get_dummies(ConcatedData)
    #filling NaN values with mean of the column
    ConcatedData = ConcatedData.fillna(ConcatedData.mean())
    #half concated data is train and lower half is for test
    X_train = ConcatedData[:train.shape[0]]
    X_test = ConcatedData[train.shape[0]:]
    y = train.SalePrice

    return X_train,X_test,y


# In[55]:


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


# In[70]:


class ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self,train,test,ytrain):
        X = train.values
        y = ytrain.values
        T = test.values
        folds = list(KFold(len(y), n_folds = self.n_folds, shuffle = True, random_state = 0))
        S_train = np.zeros((X.shape[0],len(self.base_models)))#(number of rows in X) x (number of models) 
        S_test = np.zeros((T.shape[0],len(self.base_models))) # X need to be T when do test prediction
        for i,reg in enumerate(base_models):
            print ("Fitting the base model...")
            S_test_i = np.zeros((T.shape[0],len(folds))) # X need to be T when do test prediction
            #creating folds
            for j, (train_idx,test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                #fitting and predicting specific model
                reg.fit(X_train,y_train)
                y_pred = reg.predict(X_holdout)[:]
                S_train[test_idx,i] = y_pred
                S_test_i[:,j] = reg.predict(T)[:]
            #    S_test_i[:,j] = reg.predict(X)[:]
            S_test[:,i] = S_test_i.mean(1)
        
        print ("Stacking base models...")
        param_grid = {'alpha': [1e-3,5e-3,1e-2,5e-2,1e-1,0.2,0.3,0.4,0.5,0.8,1e0,3,5,7,1e1,2e1,5e1],}
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
        grid.fit(S_train, y)
        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass
            #return best result
        y_pred = grid.predict(S_test)[:]
        return y_pred, -grid.best_score_


# In[63]:


base_models = [
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=18, max_depth=11
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0, 
            n_estimators=500, max_features=20
        ),
        GradientBoostingRegressor(
            random_state=0, 
            n_estimators=500, max_features=10, max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
        XGBRegressor(
            seed=0,
            n_estimators=500, max_depth=7,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
        ),
    ]


# In[64]:


#load training and testing data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv") 


# In[65]:


ensem = ensemble(
        n_folds=5,
	stacker=Ridge(),
        base_models=base_models
    )


# In[67]:


X_train,X_test,y_train = DataPreprocess(train,test)


# In[68]:


y_pred, score = ensem.fit_predict(X_train,X_test,y_train)

