import pandas as pd
from sklearn.model_selection import train_test_split

def create_rolling_sums(df,column,window_list):
    for window in window_list:
        df[f'{column}_sum_{window}'] = df[column].rolling(window).sum()
    return df

def merge_and_rename(level_data, rain_data):
    data = pd.merge(level_data, rain_data, on='dateTime')
    data = data.rename({'value_x':'level (m)','value_y': 'rainfall (mm)'},axis='columns')
    data = data.fillna(0)
    data['dateTime'] = pd.to_datetime(data['dateTime'])
    data.set_index('dateTime',inplace=True)
    return data

def split_data(data,rain_column,level_column):
    # X is all columns with stings containing rain_column
    X = data.filter(like=rain_column, axis=1)
    y = data[level_column]   # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    return X_train, X_test, X_val, y_train, y_test, y_val


