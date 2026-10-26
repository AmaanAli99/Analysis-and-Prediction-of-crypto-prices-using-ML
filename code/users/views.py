import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
        else:
            messages.error(request, 'Email or Mobile already existed.')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

# User Login Check View
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHome.html')
            else:
                messages.warning(request, 'Your account is not activated.')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid Login ID or Password.')
    return render(request, 'UserLogin.html')

# User Home View
def UserHome(request):
    return render(request, 'users/UserHome.html')

def train_crypto_models(request):
    # model_dir='models'
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    # data = pd.read_csv(r'C:\Users\datapoint\Desktop\cryptocurrency market price prediction\cryptocurrency_price_prediction\media\crypto_market_data.csv')
    # data['Date'] = pd.to_datetime(data['Date'])
    # data.set_index('Date', inplace=True)

    # X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    # y = data['Market Cap']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # scaler_X = MinMaxScaler()
    # X_train_scaled = scaler_X.fit_transform(X_train)
    # X_test_scaled = scaler_X.transform(X_test)

    # scaler_y = MinMaxScaler()
    # y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    # def calculate_metrics(y_true, y_pred):
    #     mse = mean_squared_error(y_true, y_pred)
    #     rmse = np.sqrt(mse)
    #     mae = mean_absolute_error(y_true, y_pred)
    #     mape = mean_absolute_percentage_error(y_true, y_pred)
    #     return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    # metrics_dict = {}

    # # Linear Regression
    # lr_model = LinearRegression()
    # lr_model.fit(X_train_scaled, y_train_scaled)
    # lr_pred = scaler_y.inverse_transform(lr_model.predict(X_test_scaled).reshape(-1, 1)).ravel()
    # metrics_dict['LinearRegression'] = calculate_metrics(y_test, lr_pred)
    # joblib.dump(lr_model, f'{model_dir}/lr_model.pkl')

    # # Random Forest
    # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf_model.fit(X_train_scaled, y_train_scaled)
    # rf_pred = scaler_y.inverse_transform(rf_model.predict(X_test_scaled).reshape(-1, 1)).ravel()
    # metrics_dict['RandomForest'] = calculate_metrics(y_test, rf_pred)
    # joblib.dump(rf_model, f'{model_dir}/rf_model.pkl')

    # # SVM
    # svm_model = SVR(kernel='rbf')
    # svm_model.fit(X_train_scaled, y_train_scaled)
    # svm_pred = scaler_y.inverse_transform(svm_model.predict(X_test_scaled).reshape(-1, 1)).ravel()
    # metrics_dict['SVM'] = calculate_metrics(y_test, svm_pred)
    # joblib.dump(svm_model, f'{model_dir}/svm_model.pkl')

    # # LSTM
    # def create_lstm_dataset(X, y, timesteps):
    #     X_lstm, y_lstm = [], []
    #     for i in range(len(X) - timesteps):
    #         X_lstm.append(X[i:i + timesteps])
    #         y_lstm.append(y[i + timesteps])
    #     return np.array(X_lstm), np.array(y_lstm)

    # timesteps = 10
    # X_train_lstm, y_train_lstm = create_lstm_dataset(X_train_scaled, y_train_scaled, timesteps)
    # X_test_lstm, y_test_lstm = create_lstm_dataset(X_test_scaled, y_test_scaled, timesteps)

    # lstm_model = Sequential([
    #     LSTM(50, activation='relu', input_shape=(timesteps, X_train_scaled.shape[1]), return_sequences=True),
    #     Dropout(0.2),
    #     LSTM(50, activation='relu'),
    #     Dropout(0.2),
    #     Dense(1)
    # ])
    # lstm_model.compile(optimizer='adam', loss='mse')
    # lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    # lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_test_lstm)).ravel()
    # metrics_dict['LSTM'] = calculate_metrics(y_test[timesteps:], lstm_pred)
    # lstm_model.save(f'{model_dir}/lstm_model.h5')

    # # Save metrics and scalers
    # pd.DataFrame(metrics_dict).T.to_csv(f'{model_dir}/model_metrics.csv')
    # joblib.dump(scaler_X, f'{model_dir}/scaler_X.pkl')
    # joblib.dump(scaler_y, f'{model_dir}/scaler_y.pkl')
    metrics_dict=pd.read_csv('models\model_metrics.csv')
    metrics_dict=metrics_dict.to_html()
    return render(request, 'users/train.html', {'metrics': metrics_dict})


from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd

def predict_market_cap(request):
    prediction = {}
    if request.method == 'POST':
        user_input = {
            'Open': float(request.POST.get('open')),
            'High': float(request.POST.get('high')),
            'Low': float(request.POST.get('low')),
            'Close': float(request.POST.get('close')),
            'Volume': float(request.POST.get('volume')),
        }

        input_df = pd.DataFrame([user_input])
        scaler_X = joblib.load('models/scaler_X.pkl')
        scaler_y = joblib.load('models/scaler_y.pkl')

        scaled_input = scaler_X.transform(input_df)

        # Load models
        lr_model = joblib.load('models/lr_model.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        svm_model = joblib.load('models/svm_model.pkl')

        # Predict
        prediction['LinearRegression'] = scaler_y.inverse_transform(
            lr_model.predict(scaled_input).reshape(-1, 1)
        ).ravel()[0]

        prediction['RandomForest'] = scaler_y.inverse_transform(
            rf_model.predict(scaled_input).reshape(-1, 1)
        ).ravel()[0]

        prediction['SVM'] = scaler_y.inverse_transform(
            svm_model.predict(scaled_input).reshape(-1, 1)
        ).ravel()[0]

    return render(request, 'users/predict.html', {'prediction': prediction})



