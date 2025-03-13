import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

def create_rolling_sums(df,column,window_list):
    for window in window_list:
        df[f'{column}_sum_{window}'] = df[column].rolling(window).sum()
    return df

def train_model(X_train, X_test, X_val, y_train, y_test, y_val,quantiles):

    models = {}
    train_predictions = {}
    test_predictions = {}
    #predictions = {f"{q*100}th_centile": [] for q in quantiles}

    for q in quantiles:
        # Train the model
        model = xgb.XGBRegressor(
            max_depth= 15,
            objective= 'reg:quantileerror',
            quantile_alpha= q,  # alpha is the quantile level
            #eval_metric= 'rmse',
            n_estimators= 8000, # Number of boosting rounds
            # Verbose logging
            device="cuda",
            learning_rate= 0.01,
            base_score = 1,
            early_stopping_rounds = 100
        )
        
        model.fit(X_train,y_train,eval_set=[(X_val, y_val)])
        #print(model.evals_result())
        models[q] = model


    
        
    return models

def make_predictions(model,x_data,quantile):
    #create new df containing just the datetime index from x_data ready to add the predictions
    df = pd.DataFrame(index=x_data.index)
    df[f"{quantile*100}th_centile"] = model.predict(x_data)
    return df