import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset;
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import os
import pickle
from tensorflow.keras.callbacks import Callback

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
SimpleRNN = tf.keras.layers.SimpleRNN
Dense = tf.keras.layers.Dense
TimeDistributed = tf.keras.layers.TimeDistributed
Bidirectional = tf.keras.layers.Bidirectional
GRU = tf.keras.layers.GRU
Attention = tf.keras.layers.Attention
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
BatchNormalization = tf.keras.layers.BatchNormalization
Input = tf.keras.layers.Input

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

os.environ["HF_DATASETS_OFFLINE"] = "1"

# %% [markdown]
# # Load datas

# %%
df = load_dataset('PleIAs/French-PD-Books', split='train[:1100]')
# {[file_id: str, ocr: int, title: str, complete_text: str, word_count: int, character_count: int]}

# # Prepare df to be {[ask: title, answer: complete_text]}
# df = df.map(lambda x: {'ask': "Genere moi une histoire qui parle de :" + x['title'], 'answer': x['complete_text']})
# df = df.filter(lambda x: x['ask'] != None and x['answer'] != None)
print(df['title'][0])
# %% [markdown]
# # Create model

# %%
# Create the model
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(df['title'])
tokenizer.fit_on_texts(df['complete_text'])

# Create the model

embedding_size = 64  # Increase this
lstm_units = 4096  # Increase this
gru_units = 2048  # Increase this
dropout_rate = 0.2
dense_units = 2048  # Increase this

# Create the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, embedding_size))
model.add(Conv1D(256, 5, activation='relu'))  # Increase the number of filters
model.add(MaxPooling1D(5))
model.add(Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)))
model.add(GRU(gru_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(BatchNormalization())
model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(LSTM(lstm_units, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


class CustomModelCheckpoint(Callback):
    def __init__(self, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.save_freq = save_freq
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.save_freq == 0:
            self.model.save(f'output/model_at_batch_{self.batch_counter}.h5')

# Instantiate the custom callback
custom_checkpoint = CustomModelCheckpoint(save_freq=100)

df = load_dataset('PleIAs/French-PD-Books', split='train')

def data_generator(batch_size):
    while True:
        for i in range(0, len(df), batch_size):
            df_batch = load_dataset('PleIAs/French-PD-Books', split=f'train[{i}:{i+batch_size}]')

            data = {'inputs': [], 'labels': []}
            for x in df_batch:
                data['inputs'].append(tokenizer.texts_to_sequences([x['title']])[0])
                data["labels"].append(tokenizer.texts_to_sequences([x['complete_text']])[0])

            seq_len = max([len(x) for x in data['inputs']])
            data['inputs'] = pad_sequences(data['inputs'], padding='post', maxlen=seq_len)
            data['labels'] = pad_sequences(data['labels'], padding='post', maxlen=seq_len)

            ds = Dataset.from_dict(data)

            tf_ds = ds.to_tf_dataset(
                columns="inputs",
                label_cols="labels",
                batch_size=batch_size,
                shuffle=False,
            )

            for inputs, labels in tf_ds:
                yield inputs, labels

# Use the data generator for model training
batch_size = 200
# model.fit(data_generator(batch_size), steps_per_epoch=len(df) // batch_size, epochs=3, verbose=1)
model.fit(data_generator(batch_size), steps_per_epoch=len(df) // batch_size, epochs=3, callbacks=[custom_checkpoint])


# %% [markdown]
# # Train model

# %%
# Train the model

# Save the model
model.save('output/model.h5')

# Save the tokenizer
with open('output/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

plot_model(model, 
           to_file='output/model.png', 
           show_shapes=True, 
           show_layer_names=True,
           show_layer_activations=True)

# Test the model
def predict(text, optimal_length=1000):
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, padding='post')
    text = model.predict(text)
    text = np.argmax(text, axis=-1)
    text = tokenizer.sequences_to_texts(text)
    text = ' '.join(text)
    return text[:optimal_length]

# Generate an history
print(predict("Genere moi une histoire qui parle de : La guerre des mondes"))