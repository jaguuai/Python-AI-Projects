"""
keras.datasets.mnist.load_data() fonksiyonu, Keras kütüphanesinin MNIST veri setini yüklemek için kullanılan bir fonksiyondur. 
MNIST, el yazısı rakamlardan oluşan bir veri setidir ve sıkça kullanılan bir derin öğrenme ve makine öğrenimi çalışmaları için bir 
başlangıç noktasıdır.
************

"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
# print(len(X_train))
# print(len(X_test))
# print(X_train[0].shape)
# 28 x 28 lik görüntüler
# print(X_train[0])
# 0 siyah noktalardır , 0-255 arası değrler beyaz noktalaar
# plt.matshow(X_train[2])
# print(y_train[2])
# print(y_train[:5])
# [5 0 4 1 9] 0-9 arası sayılar
X_train=X_train/255
# 0-1 arasında değer almak için
# Bunu yapmadan önce accrucay değerlerimiz 0.4 civarıydı 
# sonrasında 0.9 lara geldi
X_train_flattened=X_train.reshape(len(X_train),28*28)
print(X_train_flattened.shape)
X_test=X_test/255
X_test_flattened=X_test.reshape(len(X_test),28*28)
print(X_test_flattened.shape)
"""
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_train_flattened,y_train,epochs=5)

# model.evaluate() fonksiyonu, modelin test veri seti üzerindeki performansını hesaplamak için kullanılır.
print(model.evaluate(X_test_flattened,y_test))

plt.matshow(X_test[1])
y_predicted=model.predict(X_test_flattened)
print(y_predicted[1])
print(np.argmax(y_predicted[1]))

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""
"""
# Using Hidden Layer

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
print(model.evaluate(X_test_flattened,y_test))

y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

"""
# Using Flatten layer so that we don't have to call .reshape on input dataset
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test,y_test)



