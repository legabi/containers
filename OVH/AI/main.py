import sys
import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from datasets import load_dataset, Dataset;
from transformers import BertTokenizer
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Attention, Bidirectional, GRU
import pickle
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import math

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# pip install numpy datasets transformers tqdm matplotlib

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
Input = tf.keras.layers.Input
# Conv1D = tf.keras.layers.Conv1D
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# optimizer = tf.keras.optimizers.AdamW(
#         learning_rate=0.001,
#         weight_decay=0.01,
#         epsilon=1e-6,
#         global_clipnorm=1.0,  # Gradient clipping.
#     )


# optimizer = tf.keras.optimizers.Adam(
#     lr=0.001,
#     decay=0.01,
#     global_clipnorm=1.0,
# )

# if python version is >3.10, use the second, else the first
optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
        epsilon=1e-6,
        global_clipnorm=1.0,  # Gradient clipping.
    )

# optimizer = tf.keras.optimizers.Adam(
#     lr=0.001,
#     decay=0.01,
#     global_clipnorm=1.0,
# ),

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
# df = load_dataset('PleIAs/French-PD-Books', split='train[:1100]')
# {[file_id: str, ocr: int, title: str, complete_text: str, word_count: int, character_count: int]}

# # Prepare df to be {[ask: title, answer: complete_text]}
# df = df.map(lambda x: {'ask': "Genere moi une histoire qui parle de :" + x['title'], 'answer': x['complete_text']})
# df = df.filter(lambda x: x['ask'] != None and x['answer'] != None)
# print(df['title'][0])
# %% [markdown]
# # Create model
# list of all available tokenizers : https://huggingface.co/transformers/pretrained_models.html
tokenizer = BertTokenizer.from_pretrained('numind/NuNER-multilingual-v0.1', do_lower_case=True)
print(tokenizer.vocab_size)

# Create the model

# embedding_size = 64  # Increase this
# lstm_units = 4096  # Increase this
# gru_units = 2048  # Increase this
# dropout_rate = 0.2
# dense_units = 2048  # Increase this

# Create the model

dense_units = 5120
embedding_dim = 256

vocab_size = tokenizer.vocab_size

input_layer = Input(shape=(None,))

x = Embedding(vocab_size, embedding_dim)(input_layer)
x = Attention()([x, x])
x = BatchNormalization()(x)

for _ in range(32):
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Attention()([x, x])
    x = BatchNormalization()(x)

x = Dense(dense_units, activation='relu')(x)

output_layer = Dense(vocab_size, activation='softmax')(x)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

print("Num of layers: ", len(model.layers))


# plot_model(model, to_file='output/model.png', show_shapes=True)
class CustomCallback(Callback):
    def __init__(self, batch_interval=1):
        self.batch_interval = batch_interval
        self.batch_data = {'loss': [], 'accuracy': []}
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_interval == 0:
            self.batch_data['loss'].append(logs.get('loss'))
            self.batch_data['accuracy'].append(logs.get('accuracy'))
    def on_epoch_end(self, epoch, logs=None):
        plt.plot(self.batch_data['accuracy'], marker='o')
        plt.plot(self.batch_data['loss'], marker='x')
        plt.title('Model Training History')
        plt.xlabel('Batch')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.savefig(f'output/training_history_epoch_{epoch}.png')
        plt.close()


custom_callback = CustomCallback(batch_interval=100)


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
custom_checkpoint = CustomModelCheckpoint(save_freq=1000)


df = load_dataset('PleIAs/French-PD-Books', split='train', num_proc=os.cpu_count())

def encode_examples(x):
    inputs = tokenizer.encode(f"user: {x['title']} bot:", truncation=True, padding='max_length', max_length=256, add_special_tokens=True)
    labels = tokenizer.encode(f"{x['complete_text']} <ENDOFTEXT>", truncation=True, padding='max_length', max_length=256, add_special_tokens=True)

    return {'inputs': inputs, 'labels': labels}


def data_generator(batch_size):
    while True:
        for i in range(0, len(df), batch_size):
            print(f"Batch {i // batch_size + 1}/{len(df) // batch_size}")

            model.summary()

            df_batch = load_dataset('PleIAs/French-PD-Books', split=f'train[{i}:{i+batch_size}]', num_proc=os.cpu_count())
            df_batch = df_batch.map(encode_examples, num_proc=os.cpu_count())
            data = {'inputs': [], 'labels': []}

            for x in df_batch:
                data['inputs'].append(x['inputs'])
                data['labels'].append(to_categorical(x['labels'], num_classes=vocab_size))

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
batch_size = 10
# model.fit(data_generator(batch_size), steps_per_epoch=len(df) // batch_size, epochs=3, verbose=1)
# history = model.fit(data_generator(batch_size), steps_per_epoch=len(df) // batch_size, epochs=3, callbacks=[custom_checkpoint])
# do it with sessions
# history = model.fit(data_generator(batch_size), steps_per_epoch=len(df) // batch_size, epochs=3, callbacks=[custom_callback, custom_checkpoint])
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

steps_per_epoch = math.ceil(len(df) / batch_size)

with sess.as_default():
    history = model.fit(data_generator(batch_size), 
                                  steps_per_epoch=math.ceil(len(df) / batch_size),
                                  epochs=3,
                                  verbose=2,
                                  callbacks=[custom_callback, custom_checkpoint])

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.savefig('output/training_history.png')


# %% [markdown]
# # Train model

# %%
# Train the model

# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Save the model
model.save('output/model.keras')


# delete all training_history_epoch_*.png
for file in os.listdir('output'):
    if file.startswith('training_history_epoch_') and file.endswith('.png'):
        os.remove(f'output/{file}')

# Save the tokenizer
with open('output/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)