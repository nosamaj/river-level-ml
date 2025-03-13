import pandas as pd
import plotly.express as px

def plot_predictions(y_true, y_pred):
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    fig = px.scatter(df, x='Actual', y='Predicted', title='Actual vs Predicted',
                     labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
    fig.add_shape(type='line', x0=df['Actual'].min(), y0=df['Actual'].min(),
                  x1=df['Actual'].max(), y1=df['Actual'].max(),
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

def timeseries_plot(data, x, y, title, xaxis_title, yaxis_title):
    fig = px.line(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
    return fig