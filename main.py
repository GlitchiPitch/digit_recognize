from tensorflow import keras
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization

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