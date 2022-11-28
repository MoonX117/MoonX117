- 👋 Hi, I’m CHEN Xi
- 👀 My student ID is 22482784


<!---
MoonX117/MoonX117 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
### Envionment
Python 3.9

### Deployment Steps
#### Import package and prepare data
```
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
```
#### Download dataset
##### MINST Dataset
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

print('Training data: {}'.format(x_train.shape))
print('Testing data: {}'.format(x_test.shape))
```
##### FASHION MNIST Dataset
```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

print('Training data: {}'.format(x_train.shape))
print('Testing data: {}'.format(x_test.shape))
```
#### Create LeNet-5 model
```
def lenet(name='lenet'):
    model = Sequential(name=name)
    # 1st block:
    model.add(Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2nd block:
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dense layers:
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
```
#### Instantiate optimizers
```
optimizers_examples = {
    'SGD': SGD(),
    'Adagrad': Adagrad(),
    'RMSprop': RMSprop(),
    'Adam': Adam(),
    'Nadam': Nadam()
}
```
#### Training and testing
```
history_per_optimizer = dict()

print("Experiment: {0}start{1} (training logs = off)".format(log_begin_red, log_end_format))
for optimizer_name in optimizers_examples:
    tf.random.set_seed(42)
    np.random.seed(42)

    model = lenet("lenet_{}".format(optimizer_name))
    optimizer = optimizers_examples[optimizer_name]
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\t> Training with {0}: {1}start{2}".format(
        optimizer_name, log_begin_red, log_end_format))
    history = model.fit(x_train, y_train,
                        batch_size=32, epochs=10, validation_data=(x_test, y_test),
                        verbose=1)
    history_per_optimizer[optimizer_name] = history
    print('\t> Training with {0}: {1}done{2}.'.format(
        optimizer_name, log_begin_green, log_end_format))
print("Experiment: {0}done{1}".format(log_begin_green, log_end_format))
```
#### Visualize the loss function value and accuracy of training and testing
```
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='col')
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("accuracy")
ax[1, 1].set_title("val-accuracy")

lines, labels = [], []
for optimizer_name in history_per_optimizer:
    history = history_per_optimizer[optimizer_name]
    ax[0, 0].plot(history.history['loss'])
    ax[0, 1].plot(history.history['val_loss'])
    ax[1, 0].plot(history.history['accuracy'])
    line = ax[1, 1].plot(history.history['val_accuracy'])
    lines.append(line[0])
    labels.append(optimizer_name)

fig.legend(lines, labels, loc='center right', borderaxespad=0.1)
plt.subplots_adjust(right=0.85)
plt.show()
```
### Loss and accuracy curves of training and testing of MNIST dataset with different optimizers
![1](https://github.com/MoonX117/7160/blob/main/M1.png)
### Loss and accuracy curves of training and testing of FASHION MNIST dataset with different optimizers
![2](https://github.com/MoonX117/7160/blob/main/FM1.png)
