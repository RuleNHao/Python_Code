#使用mnist手写数据集实现分类


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])


# for i in np.arange(0, 2):
#     print(model(x_train[i:i+1]).numpy())
#
# print(model(x_train[:2]).numpy())

#model中的数据集形式和ndarray不同，若要寻找其中某个元素，要用x_train[i, i+1]

model.fit(x_train, y_train, epochs = 7)
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print(probability_model(x_test[0:1]).numpy())









