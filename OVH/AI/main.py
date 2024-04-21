import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset;
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import os
import pickle
from tensorflow.keras.callbacks import Callback

# test if i can write in output, else exit
if not os.path.exists('output'):
    os.makedirs('output')
with open('output/test.txt', 'w') as f:
    try:
        f.write('test')
    except:
        print('Cannot write in output')
        exit()

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

# embedding_size = 64  # Increase this
# lstm_units = 4096  # Increase this
# gru_units = 2048  # Increase this
# dropout_rate = 0.2
# dense_units = 2048  # Increase this

# Create the model
vocab_size = 1000000
embedding_dim = 512
lstm_units = 4096

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(lstm_units, return_sequences=True),
    LSTM(lstm_units, return_sequences=True),
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


class CustomModelCheckpoint(Callback):
    def __init__(self, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.save_freq = save_freq
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.save_freq == 0:
            self.model.save(f'output/model_at_batch_{self.batch_counter}.keras')

# Instantiate the custom callback
custom_checkpoint = CustomModelCheckpoint(save_freq=100)

df = load_dataset('PleIAs/French-PD-Books', split='train')

def data_generator(batch_size):
    while True:
        for i in range(0, len(df), batch_size):
            model.summary()
            df_batch = load_dataset('PleIAs/French-PD-Books', split=f'train[{i}:{i+batch_size}]')

            data = {'inputs': [], 'labels': []}
            for x in df_batch:
                data['inputs'].append(tokenizer.texts_to_sequences([x['title']])[0])
                data["labels"].append(tokenizer.texts_to_sequences([x['complete_text']])[0])

            data['inputs'] = pad_sequences(data['inputs'], padding='post', maxlen=vocab_size)
            data['labels'] = pad_sequences(data['labels'], padding='post', maxlen=vocab_size)

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
model.save('output/model.keras')

# Save the tokenizer
with open('output/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print('Model and tokenizer saved')
print('Done')
model.summary()