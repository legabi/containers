import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.pyplot as plt
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

batch_size = 8196 * 3

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def print_step(step, text=None):
    print('\r')
    print('-' * 50)
    print(f'Step {step} - {text}' if text else f'Step {step}')
    print('-' * 50)

def update_progress_bar_on_bottom(progress):
    sys.stdout.write('\r')
    progress = int(progress)
    sys.stdout.write("[%-20s] %d%%" % ('='*progress, 5*progress))

print_step(1, 'Data Loading')
update_progress_bar_on_bottom(0.8/8*10)
data = pd.read_csv('BTCUSDT.csv')
update_progress_bar_on_bottom(0.1/8*10)
data = data[['date', 'close', 'high', 'low', 'open', 'volume']]
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
update_progress_bar_on_bottom(1.5/8*10)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
update_progress_bar_on_bottom(2/8*10)

# Create training and testing datasets
print_step(2, 'Data Splitting')
update_progress_bar_on_bottom(2.5/8*10)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

class CustomCallback(Callback):
    def __init__(self, batch_interval=1):
        self.batch_interval = batch_interval
        self.batch_data = {'loss': [], 'accuracy': []}
    def on_batch_end(self, batch, logs=None):
        if batch % 1 == 0:
            print(f'Batch {batch} - Loss: {logs.get("loss")}, Accuracy: {logs.get("accuracy")} in {time.time() - start} seconds')
            data_num = batch * batch_size
            print(f'Batch {batch} - Data Number: {data_num}')
        if batch % self.batch_interval == 0:
            self.batch_data['loss'].append(logs.get('loss'))
            self.batch_data['accuracy'].append(logs.get('accuracy'))
    def on_epoch_end(self, epoch, logs=None):
        if not self.batch_data:
            return
        plt.plot(self.batch_data['accuracy'], marker='o')
        plt.title('Model Training History')
        plt.xlabel('Batch')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.savefig(f'output/training_history_epoch_{epoch}_acc.png')
        plt.close()

        plt.plot(self.batch_data['loss'], marker='o')
        plt.title('Model Training History')
        plt.xlabel('Batch')
        plt.legend(['Loss'], loc='upper left')
        plt.savefig(f'output/training_history_epoch_{epoch}_loss.png')
        plt.close()

custom_callback = CustomCallback(batch_interval=1)

class CustomModelCheckpoint(Callback):
    def __init__(self, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.save_freq = save_freq
        self.batch_counter = 0
    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.save_freq == 0:
            self.model.save(f'output/model_at_batch_{self.batch_counter}.keras')

custom_checkpoint = CustomModelCheckpoint(save_freq=1000000)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

print_step(3, 'Model Building')
update_progress_bar_on_bottom(3/8*10)

seq_length = 60
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

update_progress_bar_on_bottom(3.5/8*10)

# Reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

print_step(4, 'Model Compilation and Training')
update_progress_bar_on_bottom(4/8*10)
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, x_train.shape[2])))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
update_progress_bar_on_bottom(4.5/8*10)

start = time.time()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()
print_step(5, 'Model Training')
update_progress_bar_on_bottom(5/8*10)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, callbacks=[custom_callback, custom_checkpoint], verbose=2)
print(f'Training took {time.time() - start} seconds')
model.summary()

print_step(6, 'Model Evaluation')
update_progress_bar_on_bottom(6/8*10)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

print_step(7, 'RMSE Calculation')
update_progress_bar_on_bottom(7/8*10)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', rmse)

print_step(8, 'Model Prediction')
update_progress_bar_on_bottom(7.5/8*10)
train = data[:train_size]
valid = data[train_size:]
if len(valid) < len(predictions):
    predictions = predictions[:len(valid)]
elif len(valid) > len(predictions):
    valid = valid[:len(predictions)]
valid['Predictions'] = predictions

print_step(9, 'Model Saving')
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.savefig('output/model.png')

model.save('usdt_lstm_model.h5')

correct_predictions = 0
for i in range(1, len(predictions)):
    if predictions[i] > predictions[i-1] and y_test[i] > y_test[i-1]:
        correct_predictions += 1
    elif predictions[i] < predictions[i-1] and y_test[i] < y_test[i-1]:
        correct_predictions += 1
print(f'Correct Predictions: {correct_predictions} out of {len(predictions)}')
print(f'Percentage of Correct Predictions: {correct_predictions / len(predictions) * 100}%')
