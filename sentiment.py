import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Load IMDB reviews dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
"""
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                       metrics=['accuracy'])
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy',
                       metrics=['accuracy'])
"""
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train, partial_y_train,
                     epochs=20,
                     batch_size=512,
                     validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict)
results = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(results)
