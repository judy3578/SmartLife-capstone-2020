#학습용 코드 (ipynb): 김혜주

import pandas as pd
import urllib.request
import tkinter.messagebox
from datetime import datetime
from matplotlib.figure import Figure
import time
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./add_timestamp_data.csv')
data = df.iloc[:, 2:5]
std1 = data.std()
mean1 = data.mean()

def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict, batch_size=1):
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, 3))
    print(decoder_input.shape)

    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
            [decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)


from keras.models import load_model

fn_model = 'encoder_hello_generator.h5'
encoder_predict_model = load_model(fn_model)
fn_model = 'decoder_hello_generator.h5'
decoder_predict_model = load_model(fn_model)

fn_realtime='test_20200623_150651.csv'
df = pd.read_csv(fn_realtime)
df = df.iloc[:, 2:5]
dataRecent = np.array(df)
type(dataRecent)
dataf = dataRecent

num_steps_to_predict = 36*2
offset =20
vdata_in = np.expand_dims(dataf[offset:offset+144], 0)
vdata_out = np.expand_dims(dataf[offset+144:offset+144+num_steps_to_predict], 0)
np.shape(vdata_in),np.shape(vdata_out)
print(len(np.shape(vdata_in)))

if len(np.shape(vdata_in)) == 2:
    x_test = np.expand_dims(vdata_in, axis=0)
else:
    x_test = np.array(vdata_in)

y_test_predicted = predict(x_test, encoder_predict_model, decoder_predict_model, num_steps_to_predict)
print(y_test_predicted.shape)

vdata_in=x_test[0]
#y_test = y_test[0]
y_test_predicted = y_test_predicted[0]


t = np.arange(144+num_steps_to_predict)
n_steps_in = 144
t_in = np.arange(n_steps_in)
t_predict = t[n_steps_in:]-1
figsize = (10,5)

plt.figure(figsize=(10,5))
clr= ['C0', 'C1', 'C2']
for i in range(3):
    plt.plot(t_in, vdata_in[:,i]* std1[i] + mean1[i],color=clr[i])
    plt.plot(t_predict, vdata_out[0][:,i]* std1[i] + mean1[i], color=clr[i])
    plt.plot(t_predict, y_test_predicted[:,i]* std1[i] + mean1[i],color=clr[i], linestyle =':')
plt.title('title')
plt.show()
