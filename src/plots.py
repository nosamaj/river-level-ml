import pandas as pd
import plotly.express as px

def plot_predictions(df,y_true, y_pred):
    fig = px.scatter(df, x=y_true, y=y_pred, title='Actual vs Predicted',)
    fig.add_shape(type='line', x0=0, y0=0,
                  x1=df[y_true].max(), y1=df[y_true].max(),
                  line=dict(color='red', dash='dash'))
    return fig


# check that the columns are being applied to the right importance scores
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    print(importance)
    #map the importances to the correct feature names???

    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_importance = df_importance.sort_values(by='Importance', ascending=False)
    fig = px.bar(df_importance, x='Feature', y='Importance', title='Feature Importance',
                  labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'})
    return fig

def timeseries_plot(data_frame, y_columns, title,):
    #scattergl with lines and no markers
    df_sorted = data_frame.sort_index()
    fig = px.line(df_sorted, x=df_sorted.index, y=y_columns, title=title)
    return fig