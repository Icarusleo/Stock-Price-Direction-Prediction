# Stock Price Direction Prediction

This project aims to predict the daily price movement direction (up or down) of stocks (e.g., GOOG, MRK) using historical stock data and financial news headlines, utilizing deep learning-based approaches.

In this project, not only price and volume information but also technical indicators and Natural Language Processing (NLP)-based sentiment analysis are combined to develop **Multimodal** models.

## 🚀 Features

- **Time Series Analysis:** Fetching historical stock data (OHLCV) using the YFinance library.
- **Technical Indicators (pandas_ta):** Adding features like RSI, MACD, Bollinger Bands (BBANDS), ATR, and Simple Moving Averages (SMA) to the dataset, and making the data stationary (Log-returns).
- **Natural Language Processing (FinBERT):** Integrating daily financial news headlines by embedding them using the HuggingFace `ProsusAI/finbert` model.
- **Advanced Models:**
  - **LSTM** and **Bi-LSTM** models using only numerical (technical) data.
  - Hybrid **FinBERT + CNN + LSTM** architecture combining numerical and text data.
  - Hybrid **FinBERT + Bi-LSTM** architecture, which yielded the best results.
- **Walk-Forward Validation:** Testing model robustness using the Time Series Split cross-validation method suitable for time series.

## 🛠️ Installation and Requirements

To run the project in your local environment or on Google Colab, the following libraries must be installed:

```bash
pip install tensorflow transformers yfinance pandas numpy pandas_ta scikit-learn matplotlib seaborn
```

*Note: PyTorch and Transformers libraries are downloaded in the background to use the FinBERT model.*

## 📊 Dataset Preparation

1. **Price Data:** Stock data for specific date ranges are downloaded via `yfinance`.
2. **Technical Indicators:** RSI, MACD, etc., are calculated using the `pandas_ta` library.
3. **Labeling (Target):** If the next day's closing price is higher than today's closing price, it is labeled as `1` (Up); if lower, `0` (Down).
4. **News Data:** The analyst/news dataset in CSV format is filtered by stock and merged on a daily basis.

## 🧠 Models and Performance Evaluation

Model performances were evaluated using Accuracy, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC) metrics. Visualizations are provided via the Confusion Matrix.

- **Bi-LSTM Model:** Yielded one of the most consistent and best results on the 5-year extended dataset with technical indicators.
- **FinBERT + LSTM/Bi-LSTM (Hybrid):** An advanced structure combining the NLP (text) and technical (numerical) branches was used to model the relationship between market sentiment and trends.

## 📌 Usage

To run the project:
1. Open the Notebook file (`.ipynb`) using Jupyter Notebook or Google Colab.
2. If you are using data from Google Drive, mount your Drive in the relevant cell (`drive.mount`).
3. Update the file paths of the datasets according to your own directory structure.
4. Run the cells sequentially to fetch the data, complete preprocessing, and train the models.

## 🤝 Contributing

This project is open to contributions. If you find any bugs or have suggestions for improvement, please open an **Issue** or submit a **Pull Request**.

## 📄 License

This project is licensed under the MIT License.
