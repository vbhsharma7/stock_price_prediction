import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import plotly
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import array

start_date = '2020-01-01'
end_date = '2023-06-15'

st.title('Stock Price Prediction')
ticker = st.text_input("Enter Stock Ticker",'NMDC.NS')

df = yf.download(ticker, start=start_date, end=end_date)

# Describe dataset
st.subheader('Data from 2020-2023')
st.write(df.describe())

#visualization 
st.subheader('Closing Price v/s Time Chart')
Closed_price = df['Close']
fig, ax = plt.subplots(figsize=(12, 4))
Closed_price.plot(label=ticker)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Moving Average v/s Time Chart')
plt.figure(figsize=(12, 4))
df['Close'].rolling(window=120).mean().plot(label='120-day Moving Average')
df['Close'].plot(label=ticker)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt.gcf()) 

# splitting the data into X_train and Y_train and scaling the dataset
st.subheader('120 Days Prediction v/s Time Chart')
df1=df.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=load_model('C:/Users/knksh/Desktop/stock_price model/keras_model.h5')

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

x_input=test_data[len(test_predict)-100:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=200
i=0
while(i<120):

    if(len(temp_input)>100):
        # print(temp_input)
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        # x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

# print(lst_output)
day_new=np.arange(1,len(x_input[0])+1)
day_pred=np.arange(len(x_input[0])+1,len(x_input[0])+121)

plt.figure(figsize=(12, 4))
plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-len(x_input[0]):]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(plt.gcf())





