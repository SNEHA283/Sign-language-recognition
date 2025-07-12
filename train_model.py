import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import cv2

train_path = 'gesture/train'
test_path = 'gesture/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(train_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(test_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)),
    MaxPool2D(2,2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPool2D(2,2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005),
    EarlyStopping(monitor='val_loss', patience=2)
]

history = model.fit(train_batches, epochs=10, callbacks=callbacks, validation_data=test_batches)
model.save('best_model_dataflair3.h5')
print("Model saved.")
