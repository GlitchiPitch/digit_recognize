from tensorflow import keras
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

model = keras.Sequential([
    Input(shape=(28,28,1)),
    Flatten(),
    Dense(300, activation='relu'),
    # Dropout(.8), BatchNormalization
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=.2)

his = model.fit(
    x_train, y_train_cat, batch_size=32, epochs=5, validation_data=(x_val_split, y_val_split)
)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()

model.evaluate(x_test, y_test_cat)

n = 2
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

mask = pred == y_test
x_false = x_test[~mask]
p_false = pred[~mask]

for i in range(5):
  plt.imshow(x_false[i], cmap=plt.cm.binary)
  plt.show()