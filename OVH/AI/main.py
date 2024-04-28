import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset;
from transformers import BertTokenizer
from keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import os
import pickle
from tensorflow.keras.callbacks import Callback
import tqdm


# test if i can write in output, else exit
if not os.path.exists('output'):
    os.makedirs('output')
with open('output/test.txt', 'w') as f:
    try:
        f.write('test')
    except:
        print('Cannot write in output')
        exit()

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
Dense = tf.keras.layers.Dense
TimeDistributed = tf.keras.layers.TimeDistributed
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the model

# embedding_size = 64  # Increase this
# lstm_units = 4096  # Increase this
# gru_units = 2048  # Increase this
# dropout_rate = 0.2
# dense_units = 2048  # Increase this

# Create the model
vocab_size = tokenizer.vocab_size
embedding_dim = 4096
lstm_units = 4096

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(LSTM(lstm_units))
model.add(Dropout(0.2))
for _ in range(32):
    model.add(Dense(4096, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

print(f'The model has {sum(p.numel() for p in model.trainable_variables)} parameters')

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
                inputs = tokenizer.encode(x['title'], return_tensors='tf', truncation=True, padding='max_length', max_length=vocab_size)
                labels = tokenizer.encode(x['complete_text'], return_tensors='tf', truncation=True, padding='max_length', max_length=vocab_size)
                data['inputs'].append(inputs)
                data['labels'].append(labels)

            data['inputs'] = tf.concat(data['inputs'], axis=0)
            data['labels'] = tf.concat(data['labels'], axis=0)

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
batch_size = 100
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