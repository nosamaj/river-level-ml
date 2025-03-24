import streamlit as st
import pandas as pd
from data_preprocessing import create_rolling_sums, merge_and_rename, split_data
from model import train_model, make_predictions
from plots import plot_predictions, plot_feature_importance, timeseries_plot
from utils import load_csv


def main():
    
	#defualt to wide mode
    st.set_page_config(layout="wide")

    default_windows = ["1h", "2h", "4h", "8h", "24h", "7d", "30d", "90d"]
    quantiles = [0.05, 0.50, 0.95]

    default_rain_column = "rainfall (mm)"
    default_level_column = "level (m)"

    st.title("Level and Rain Data Model Training")

    # File upload for level data
    level_file = st.file_uploader("Upload Level CSV", type=["csv"])
    # File upload for rain data
    rain_file = st.file_uploader("Upload Rain CSV", type=["csv"])

    if level_file is not None and rain_file is not None:
        # Load data
        level_data = load_csv(level_file)
        rain_data = load_csv(rain_file)

        # Merge and rename columns
        data = merge_and_rename(level_data, rain_data)

        # Create rolling sums
        data = create_rolling_sums(data, default_rain_column, default_windows)

        # Split data
        X_train, X_test, X_val, y_train, y_test, y_val = split_data(
            data, default_rain_column, default_level_column
        )

        # Train model
        models = train_model(X_train, X_test, X_val, y_train, y_test, y_val, quantiles)

        # Prepare results
        df_results = X_test.copy().join(y_test)

        for quantile in quantiles:
            df_predictions = make_predictions(models[quantile], X_test, quantile)
            df_results = pd.merge(
                df_results,
                df_predictions,
                how="outer",
                left_index=True,
                right_index=True,
            )

        # Create columns for layout
        col1, col2 = st.columns(2)

        # Left column: Display dataframes
        with col1:
            st.subheader("Level Data")
            if level_data is not None:
                st.write(level_data.head())
            st.subheader("Rain Data")
            if rain_data is not None:
                st.write(rain_data.head())
            st.subheader("Model Results")
            st.write(df_results)

        # Right column: Display plots
        with col2:
            st.subheader("Plots")
            st.write(
                plot_predictions(
                    df_results,
                    default_level_column,
                    [
                        f"{quantiles[0]*100}th_centile",
                        f"{quantiles[1]*100}th_centile",
                        f"{quantiles[2]*100}th_centile",
                    ],
                )
            )
            st.write(plot_feature_importance(models[quantile], X_train.columns))
            st.write(
                timeseries_plot(
                    df_results,
                    [
                        f"{quantiles[0]*100}th_centile",
                        f"{quantiles[1]*100}th_centile",
                        f"{quantiles[2]*100}th_centile",
                        default_level_column,
                    ],
                    "Level Data",
                )
            )


if __name__ == "__main__":
    main()
