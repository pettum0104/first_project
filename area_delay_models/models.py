import numpy as np
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


area_xgboost_model = XGBRegressor(objective='reg:squarederror', tree_method="hist", eval_metric=mape, n_jobs=-1, 
                     n_estimators=100, reg_lambda=10, max_depth=4, device="cuda")
delay_dtr_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=400)

def get_models():
    return area_xgboost_model, delay_dtr_model
