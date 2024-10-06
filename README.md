# Time-Series Stock Price Prediction Using ARIMA and LSTM Models

This program predicts stock prices using two different machine learning models: ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory). The dataset is preprocessed, and various features are engineered before training the models. The ARIMA model predicts based on historical stock prices, while the LSTM model uses time-series data to predict future stock prices. The results are visualized, and the performance of both models is compared.

## Overview
Features of the Program:
### Data Preprocessing:

Handling missing values using forward fill.
Conversion of stock volume values into numerical format (e.g., "M" for millions, "B" for billions).
Feature engineering, including the creation of moving averages (MA_50, MA_200) and lag features.
### Models:

ARIMA: A classical time-series forecasting method trained on the closing stock price.
LSTM: A neural network model designed to handle sequential data, which is used for predicting future closing prices based on the previous 60 days.
### Evaluation:

Both ARIMA and LSTM models are evaluated using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
Visualizations of both models' predictions vs. actual values are provided for easy comparison.
## Prerequisites
Ensure you have the following dependencies installed:

bash
Copy code
pip install numpy pandas matplotlib seaborn tensorflow statsmodels scikit-learn
You also need Python 3.7 or higher.

Running the Program
Clone or download the repository to your local machine:

```(bash)
git clone https://github.com/yourusername/time-series-stock-prediction.git
```

Navigate to the project directory:

```(bash)
cd time-series-stock-prediction
```

Prepare your dataset:

Place your stock price data in the csv format in the root directory of the project.
Make sure the CSV file includes the following columns: date, closing price, open price, high price, low price, and volume.
Update file paths:

In file.py, update the file_path variable to point to your dataset:
```(python)
file_path = 'path/to/your/stock_price.csv'
```

Run the script:

```(bash)
python file.py
```

Explanation of Key Sections
Data Preprocessing: The program processes the data by handling missing values, converting the volume into a numeric format, and generating moving averages and lag features for use in the models.

ARIMA Model:

Trained on the closing price of the stock.
Provides predictions for future stock prices.
Uses a simple (5,1,0) configuration for ARIMA, but you may tune these parameters for better accuracy.
LSTM Model:

Neural network model designed for sequential data.
Reshapes the training and test data into 3D format to fit the LSTM model.
Uses 50 LSTM units with a 20% dropout to prevent overfitting.
Visualization:

Both ARIMA and LSTM predictions are visualized alongside the actual stock prices to provide an intuitive comparison.
The program automatically generates and displays plots of the predictions vs. actual values.
Evaluation:

The program calculates and prints the RMSE and MAE for both the ARIMA and LSTM models, providing insights into their performance.
Troubleshooting
If you encounter an issue with TensorFlow, ensure you are using a compatible version of TensorFlow (>= 2.12.0).
Ensure that the dataset is clean and contains no unexpected symbols in numeric columns.
Example Dataset Format
Your dataset (CSV file) should look like this:

date	closing price	open price	high price	low price	volume

2020-01-01	150.5	148.2	152.0	147.5	1.2M

2020-01-02	153.0	150.0	154.0	149.5	1.5M
### Author
Your Name : Hemang Vijay Borse

GitHub: https://github.com/master-Hemang
