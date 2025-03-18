import streamlit as st
import pandas as pd
from data_preprocessing import create_rolling_sums, merge_and_rename, split_data
from model import train_model,make_predictions
from plots import plot_predictions, plot_feature_importance, timeseries_plot
from utils import load_csv

def main():

    default_windows = ['1h','2h','4h','8h','24h','7d','30d','90d']
    quantiles = [0.05, 0.50, 0.95]

    default_rain_column = 'rainfall (mm)'
    default_level_column = 'level (m)'

    st.title("Level and Rain Data Model Training")

    # File upload for level data
    level_file = st.file_uploader("Upload Level CSV", type=["csv"])
    # File upload for rain data
    rain_file = st.file_uploader("Upload Rain CSV", type=["csv"])

    if level_file is not None and rain_file is not None:
        # Load data
        level_data = load_csv(level_file)
        rain_data = load_csv(rain_file)

        # Display data
        st.subheader("Level Data")
        st.write(level_data.head())
        st.subheader("Rain Data")
        st.write(rain_data.head())

        # Merge and rename columns
        data = merge_and_rename(level_data, rain_data)

        # Create rolling sums
        data = create_rolling_sums(data,default_rain_column,default_windows)

        # Split data
        X_train, X_test, X_val, y_train, y_test, y_val = split_data(data,default_rain_column,default_level_column)
                                                                    

        # Train model
        models = train_model(X_train, X_test, X_val, y_train, y_test, y_val,quantiles)

        # Display plots
        st.subheader("Model Results")

        
        for quantile in quantiles:
            df_predictions = make_predictions(models[quantile],X_test,quantile)
            st.write(f"Quantile: {quantile}")
            st.write(plot_predictions(y_test, df_predictions[f"{quantile*100}th_centile"]))
            st.write(plot_feature_importance(models[quantile], X_train.columns))
            st.dataframe(df_predictions)
            st.write(timeseries_plot(df_predictions, [f"{quantile*100}th_centile"], "Level Data"))    

if __name__ == "__main__":
    main()