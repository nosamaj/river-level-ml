# Streamlit Level and Rain Data Analysis App

This project is a Streamlit application designed to allow users to upload CSV files containing level and rain data, train a machine learning model, and visualize the results through interactive plots.

## Project Structure

```
streamlit-app
├── src
│   ├── app.py          # Main entry point of the Streamlit application
│   ├── model.py        # Functions and classes for model training and evaluation
│   ├── plots.py        # Functions to generate and display plots
│   └── utils.py        # Utility functions for data processing
├── requirements.txt     # List of dependencies required for the project
└── README.md            # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd streamlit-app
   ```

2. **Install the required packages:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```bash
   pip install -r requirements.txt
   ```

## Usage Guidelines

1. **Run the Streamlit application:**
   After installing the dependencies, you can start the application by running:
   ```bash
   streamlit run src/app.py
   ```

2. **Upload CSV files:**
   The application will prompt you to upload two CSV files: one for level data and another for rain data.

3. **Model Training:**
   Once the files are uploaded, the application will train a machine learning model using the provided data.

4. **View Plots:**
   After training, the application will display interactive plots based on the model's predictions and the input data.

## Application Functionality

- **File Upload:** Users can upload their level and rain data in CSV format.
- **Model Training:** The application trains a machine learning model using the uploaded data.
- **Interactive Visualizations:** Users can view plots generated from the model's predictions and the input data, providing insights into the relationship between level and rain.

## Dependencies

The application requires the following Python packages:

- Streamlit
- pandas
- xgboost
- plotly
- scikit-learn

Make sure to install these packages as specified in `requirements.txt`.