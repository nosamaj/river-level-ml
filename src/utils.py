import pandas as pd

def load_csv(file):
    """Load a CSV file into a pandas DataFrame."""
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def preprocess_data(data):
    """Preprocess the data as needed for model training."""
    # Example preprocessing steps
    # This function can be customized based on the specific requirements of the data
    data = data.dropna()  # Remove missing values
    return data



def split_data(data, target_column):
    """Split the data into features and target variable."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y