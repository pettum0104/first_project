from xgboost import XGBRegressor


model = XGBRegressor(random_state=42, objective='reg:absoluteerror', tree_method="hist", 
                     eval_metric=mape, n_jobs=-1, n_estimators=100,
                     learning_rate=0.1, reg_lambda=10,
                     reg_alpha=10, max_depth=5, device="cuda")

def get_model():
    return model
