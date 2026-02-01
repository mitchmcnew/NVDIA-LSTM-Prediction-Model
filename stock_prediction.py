import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from datetime import date

from keras.layers import Dense, Dropout, InputLayer, LSTM
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import kagglehub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os


# to create windows for LSTM to look at
def windows(features, labels, window_size=60):
    x, y = [], []
    for i in range(window_size, len(features)):
        x.append(features[i - window_size:i])
        y.append(labels[i])
    return np.array(x), np.array(y)


def main():
    # obtain wanted data
    today = date.today().strftime('%Y-%m-%d')
    tickers = ['NVDA', 'VTI', 'BIL', 'IWM', 'SPY', 'IVE', 'IVW', 'QUAL', 'MTUM', 'VIG', 'ARKK']

    print("Downloading Market Data...")
    df = yf.download(tickers, start="2015-12-31", end=today, auto_adjust=True)

    # to track volume change of NVDIA
    vol_nvda = df['Volume']['NVDA']
    vol_feature = vol_nvda.pct_change().dropna()

    price_df = df['Close']
    returns = price_df.pct_change().dropna()
    ff_features = pd.DataFrame(index=returns.index, columns=returns.columns)

    # Feature Calculations
    ff_features['mk_rf'] = returns['VTI'] - returns['BIL']
    ff_features['SMB'] = returns['IWM'] - returns['SPY']
    ff_features['HML'] = returns['IVE'] - returns['IVW']
    ff_features['RMW'] = returns['QUAL'] - returns['MTUM']
    ff_features['CMA'] = returns['VIG'] - returns['ARKK']
    ff_features['VOl_Change'] = vol_feature.reindex(ff_features.index)

    # discard the original ticker columns
    ff_features = ff_features[['mk_rf', 'SMB', 'HML', 'RMW', 'CMA', 'VOl_Change']]

    # NLP processing
    analyzer = SentimentIntensityAnalyzer()

    print("Loading Dataset 1 (Foundation)...")
    path1 = kagglehub.dataset_download("dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024")
    files1 = [f for f in os.listdir(path1) if f.endswith('.csv')]
    df_foundation = pd.read_csv(os.path.join(path1, files1[0]))


    df_foundation.columns = df_foundation.columns.str.lower()

    text_col = 'title'

    df_foundation['date'] = pd.to_datetime(df_foundation['date'])

    # Filter first
    df_market_news = df_foundation[df_foundation['date'] >= '2015-12-31'][['date', text_col]]

    # Aggregate and Score
    df_daily_market = df_market_news.groupby('date')[text_col].apply(' '.join).reset_index()
    df_daily_market['market_sentiment'] = df_daily_market[text_col].apply(
        lambda x: analyzer.polarity_scores(x)['compound'])

    print("Loading Dataset 2 (Deep Dive)...")
    path2 = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
    news_df = pd.read_csv(os.path.join(path2, "analyst_ratings_processed.csv"))

    #Standardize columns
    news_df.columns = news_df.columns.str.lower()
    nvda_specific_news = news_df[news_df['stock'] == 'NVDA'].copy()

    #Handle mixed timezones, then remove timezone info
    nvda_specific_news['date'] = pd.to_datetime(nvda_specific_news['date'], utc=True).dt.tz_localize(
        None).dt.normalize()

    # Auto-detect text column
    text_col_2 = next((c for c in nvda_specific_news.columns if c in ['headline', 'headlines', 'news', 'title']),
                      'title')

    df_daily_ticker = nvda_specific_news.groupby('date')[text_col_2].apply(' '.join).reset_index()
    df_daily_ticker['ticker_sentiment'] = df_daily_ticker[text_col_2].apply(
        lambda x: analyzer.polarity_scores(x)['compound'])

    # Filler
    print("Loading Dataset 3 (2025 Filler)...")
    path3 = kagglehub.dataset_download("pratyushpuri/financial-news-market-events-dataset-2025")
    files3 = [f for f in os.listdir(path3) if f.endswith('.csv')]
    df_filler = pd.read_csv(os.path.join(path3, files3[0]))

    # Standardize column name
    df_filler.columns = df_filler.columns.str.lower()

    sent_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    # Ensure values are lowercased before mapping to match our key
    df_filler['filler_sent'] = df_filler['sentiment'].str.lower().map(sent_map)
    df_filler['date'] = pd.to_datetime(df_filler['date'])
    df_daily_filler = df_filler.groupby('date')['filler_sent'].mean().reset_index()


    print("Merging Data...")
    # Merge all into your main features dataframe
    ff_features = ff_features.reset_index().rename(columns={'Date': 'date'})  # Ensure Date is a column
    ff_features['date'] = pd.to_datetime(ff_features['date'])

    # Progressive Merges - using on='date' since we standardized everything
    ff_features = pd.merge(ff_features, df_daily_market[['date', 'market_sentiment']], on='date', how='left')
    ff_features = pd.merge(ff_features, df_daily_ticker[['date', 'ticker_sentiment']], on='date', how='left')
    ff_features = pd.merge(ff_features, df_daily_filler[['date', 'filler_sent']], on='date', how='left')

    # Data Cleaning: Forward fill missing sentiment days (common on weekends/low news days)
    ff_features[['market_sentiment', 'ticker_sentiment', 'filler_sent']] = ff_features[
        ['market_sentiment', 'ticker_sentiment', 'filler_sent']].fillna(method='ffill').fillna(0)

    # FIX: Set index back to date to align with returns, then create ylabel
    ff_features = ff_features.set_index('date')

    # Now that indices align (Date to Date), we can create the label
    ylabel = returns.reindex(ff_features.index)['NVDA']

    # Drop NaNs created by alignment issues (if any)
    valid_indices = ylabel.dropna().index
    ff_features = ff_features.loc[valid_indices]
    ylabel = ylabel.loc[valid_indices]

    print(f"Final Feature Shape: {ff_features.shape}")

    # split into temp and test
    features_temp, features_test, labels_temp, labels_test = train_test_split(ff_features, ylabel,
                                                                              test_size=0.1, shuffle=False)

    # create train and validation
    features_train, features_val, labels_train, labels_val = train_test_split(features_temp, labels_temp,
                                                                              test_size=0.11, shuffle=False)

    # Standardize features
    ct = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(),
             ['mk_rf', 'SMB', 'HML', 'RMW', 'CMA', 'market_sentiment', 'ticker_sentiment', 'filler_sent']),
            ('vol', PowerTransformer(), ['VOl_Change']),
        ]
    )

    # fit/transform to train val and test
    features_train_scale = ct.fit_transform(features_train)
    features_val_scale = ct.transform(features_val)
    features_test_scale = ct.transform(features_test)

    # scale the labels
    scalar = StandardScaler()
    y_train_scale = scalar.fit_transform(labels_train.values.reshape(-1, 1))
    y_val_scale = scalar.transform(labels_val.values.reshape(-1, 1))
    y_test_scale = scalar.transform(labels_test.values.reshape(-1, 1))

    # incorporate windows
    X_train_3D, y_train_3D = windows(features_train_scale, y_train_scale)
    X_val_3D, y_val_3D = windows(features_val_scale, y_val_scale)
    X_test_3D, y_test_3D = windows(features_test_scale, y_test_scale)

    # Initialize Model
    my_model = Sequential()

    # input layer
    my_model.add(InputLayer(input_shape=(60, ff_features.shape[1])))
    my_model.add(LSTM(64, return_sequences=True))
    my_model.add(Dropout(0.3))
    my_model.add(LSTM(32, return_sequences=False))
    my_model.add(Dropout(0.3))
    my_model.add(Dense(16, activation='relu'))
    my_model.add(Dense(1))

    # initialize optimizer
    opt = Adam(learning_rate=0.0001)

    # early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    # compile model
    my_model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    print("Training Model...")
    history = my_model.fit(X_train_3D, y_train_3D, epochs=100, batch_size=32,
                           verbose=1, validation_data=(X_val_3D, y_val_3D), callbacks=[early_stop])

    res_mse, res_mae = my_model.evaluate(X_test_3D, y_test_3D)
    print(f"MSE: {res_mse}, MAE: {res_mae}")

    # analyze by percentage points
    y_pred_scaled = my_model.predict(X_test_3D)

    # Invert the scaling
    y_pred_real = scalar.inverse_transform(y_pred_scaled)
    y_test_real = scalar.inverse_transform(y_test_3D)

    real_mae = np.mean(np.abs(y_pred_real - y_test_real))

    # Print the "Percentage Error"
    print(f"Final Model Accuracy: On average, the model is off by {real_mae * 100:.2f}% per day.")

    # 4. Predict Tomorrow's Return
    latest_window = features_test_scale[-60:]

    # Reshape to (1, 60, 9) -> (1 sample, 60 days, 9 features)
    latest_window_3D = latest_window.reshape(1, 60, ff_features.shape[1])

    # Make the prediction
    prediction_scaled = my_model.predict(latest_window_3D)

    # Inverse transform to get the actual percentage return
    prediction_real = scalar.inverse_transform(prediction_scaled)

    # Output the result
    predicted_return = prediction_real[0][0]
    print(f"\nPredicted NVDIA return for tomorrow: {predicted_return * 100:.2f}%")

    # 5. Plot the Training History
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Fama-French LSTM: Model Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

main()