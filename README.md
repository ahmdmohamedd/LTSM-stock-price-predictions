# Stock Market Price Prediction using LSTM

This project contains a project that predicts stock market prices using a Long Short-Term Memory (LSTM) model. The project utilizes historical stock price data and predicts future stock prices for the next 30 days. The model has been built using TensorFlow and Keras libraries.

## Project Overview

The project involves the following key steps:

1. **Data Collection**: Historical stock price data is collected from Yahoo Finance.
2. **Data Preprocessing**: The data is scaled, split into training and testing sets, and reshaped for use in an LSTM model.
3. **Model Building**: An LSTM model is constructed using TensorFlow/Keras to learn from the data.
4. **Training and Testing**: The model is trained on the training data and evaluated on the test data.
5. **Future Predictions**: The model predicts stock prices for the next 30 days, and the results are visualized.
6. **Visualization**: The stock price predictions are plotted alongside the actual stock prices for better visualization and comparison.

## Requirements

To run this project, you will need the following libraries:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow/Keras
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

## Dataset

The historical stock price data is obtained from Yahoo Finance. You can choose any stock's data and download it as a CSV file. This dataset contains features like the **Open**, **High**, **Low**, **Close**, **Volume**, and **Adj Close** prices for each day.

## Project Structure

```
.
├── stock_price_prediction_lstm.ipynb   # Jupyter notebook with the entire code
├── README.md                           # Project documentation
├── data                                # Directory containing the dataset (CSV file)
│   └── stock_data.csv                  # Stock price dataset (replace with your data)
└── images                              # Directory containing visualizations
```

## Model Architecture

The LSTM model consists of the following layers:
- Two LSTM layers with 50 units each.
- A Dense layer with 25 units.
- A final Dense layer with 1 unit to output the stock price.

```python
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Build the LSTM model with Input layer
inputs = Input(shape=(time_step, 1))
x = LSTM(units=50, return_sequences=True)(inputs)
x = LSTM(units=50, return_sequences=False)(x)
x = Dense(units=25)(x)
outputs = Dense(units=1)(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mean_squared_error')
```

## Data Preprocessing

The historical stock prices are normalized using `MinMaxScaler` to ensure that all the values lie between 0 and 1. This helps the LSTM model to converge faster and perform better.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
```

The dataset is split into training and testing sets, where the training set is used to fit the model and the testing set is used to evaluate its performance.

## Model Training and Testing

The model is trained using the training data for a specified number of epochs and a batch size. After training, the model's performance is evaluated by comparing the predicted stock prices with the actual stock prices from the test data.

```python
# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)

# Evaluate model on test data
test_predictions = model.predict(X_test)
```

## Future Predictions (Next 30 Days)

The model can be used to predict stock prices for the next 30 days. Using the last few data points from the test set, we generate predictions for the next 30 days and plot them alongside the actual historical prices.

```python
# Predict future stock prices for the next 30 days
n_days = 30
x_input = test_data[-time_step:].reshape(1, -1)
predictions = []

for i in range(n_days):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    predictions.append(yhat[0][0])
```

## Results

### Visualization

The model's predictions for the test data are plotted along with the actual stock prices. Additionally, the predictions for the next 30 days are displayed in a different color for better visibility.

### Key Findings
- The model performs well in capturing the trends of the stock prices.
- The predicted prices for the next 30 days are visualized to show the potential stock movement.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/LTSM-stock-price-predictions.git
   ```

2. Navigate to the project directory:
   ```bash
   cd LTSM-stock-price-predictions
   ```

3. Run the Jupyter notebook to train the model and make predictions:
   ```bash
   jupyter notebook LTSM_stock_price_predictions.ipynb
   ```

4. Replace the stock data in the `data` folder with your own dataset if needed.

## Conclusion

This project demonstrates how to use LSTM models for time series forecasting, specifically for predicting stock market prices. The model successfully captures historical trends and can be used to predict future stock prices with reasonable accuracy.

Feel free to explore the project and modify the dataset to try different stock price predictions.
