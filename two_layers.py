# coding: utf-8
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# Load MNIST dataset from keras
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Flatten the images and normalize
X_train_full = X_train_full.reshape(-1, 28 * 28).astype("float32")
X_test = X_test.reshape(-1, 28 * 28).astype("float32")

X_train_full = ((X_train_full / 255.) - .5) * 2
X_test = ((X_test / 255.) - .5) * 2

# Split training set into 70% training and 30% test set
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=123, stratify=y_train_full
)

print(X_train.shape)
print(y_train.shape)

# Visualize the first digit of each class:
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Visualize 25 different versions of "7":
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Optional: free up some memory by deleting non-used arrays
del X_train_full, y_train_full

##########################
### MODEL
##########################

def sigmoid(z):                                        
    return 1. / (1. + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

def compute_macro_auc(model, X, y, num_labels=10, minibatch_size=100):
    y_true, y_probas = [], []
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for features, targets in minibatch_gen:
        _, _, probas = model.forward(features)
        y_true.extend(targets)
        y_probas.extend(probas)
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    y_onehot = int_to_onehot(y_true, num_labels=num_labels)
    return roc_auc_score(y_onehot, y_probas, average="macro", multi_class="ovr")

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden1, num_hidden2, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        self.weight_h1 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden1, num_features))
        self.bias_h1 = np.zeros(num_hidden1)
        self.weight_h2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden2, num_hidden1))
        self.bias_h2 = np.zeros(num_hidden2)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden2))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h1 = np.dot(x, self.weight_h1.T) + self.bias_h1
        a_h1 = sigmoid(z_h1)
        z_h2 = np.dot(a_h1, self.weight_h2.T) + self.bias_h2
        a_h2 = sigmoid(z_h2)
        z_out = np.dot(a_h2, self.weight_out.T) + self.bias_out
        a_out = softmax(z_out)
        return a_h1, a_h2, a_out

    def backward(self, x, a_h1, a_h2, a_out, y):
        y_onehot = int_to_onehot(y, self.num_classes)
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        delta_out = d_loss__d_a_out
        d_z_out__dw_out = a_h2
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        d_z_out__a_h2 = self.weight_out
        d_loss__a_h2 = np.dot(delta_out, d_z_out__a_h2)
        d_a_h2__d_z_h2 = a_h2 * (1. - a_h2)
        delta_h2 = d_loss__a_h2 * d_a_h2__d_z_h2
        d_z_h2__d_w_h2 = a_h1
        d_loss__d_w_h2 = np.dot(delta_h2.T, d_z_h2__d_w_h2)
        d_loss__d_b_h2 = np.sum(delta_h2, axis=0)
        d_z_h2__a_h1 = self.weight_h2
        d_loss__a_h1 = np.dot(delta_h2, d_z_h2__a_h1)
        d_a_h1__d_z_h1 = a_h1 * (1. - a_h1)
        delta_h1 = d_loss__a_h1 * d_a_h1__d_z_h1
        d_z_h1__d_w_h1 = x
        d_loss__d_w_h1 = np.dot(delta_h1.T, d_z_h1__d_w_h1)
        d_loss__d_b_h1 = np.sum(delta_h1, axis=0)
        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h2, d_loss__d_b_h2, d_loss__d_w_h1, d_loss__d_b_h1)

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size=100)
        for X_train_mini, y_train_mini in minibatch_gen:
            a_h1, a_h2, a_out = model.forward(X_train_mini)
            gradients = model.backward(X_train_mini, a_h1, a_h2, a_out, y_train_mini)
            model.weight_out -= learning_rate * gradients[0]
            model.bias_out -= learning_rate * gradients[1]
            model.weight_h2 -= learning_rate * gradients[2]
            model.bias_h2 -= learning_rate * gradients[3]
            model.weight_h1 -= learning_rate * gradients[4]
            model.bias_h1 -= learning_rate * gradients[5]
        train_loss = mse_loss(y_train, model.forward(X_train)[2])
        train_acc = accuracy(y_train, np.argmax(model.forward(X_train)[2], axis=1))
        valid_loss = mse_loss(y_valid, model.forward(X_valid)[2])
        valid_acc = accuracy(y_valid, np.argmax(model.forward(X_valid)[2], axis=1))
        epoch_loss.append(train_loss)
        epoch_train_acc.append(train_acc * 100)
        epoch_valid_acc.append(valid_acc * 100)
        print(f"Epoch: {e+1:03d}/{num_epochs:03d} | Train MSE: {train_loss:.2f} | Train Acc: {train_acc * 100:.2f}% | Valid Acc: {valid_acc * 100:.2f}%")
    return epoch_loss, epoch_train_acc, epoch_valid_acc

# Instantiate and train the model
model = NeuralNetMLP(num_features=28*28, num_hidden1=500, num_hidden2=500, num_classes=10)
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid, num_epochs=20, learning_rate=0.1
)

# Plot results
plt.plot(range(len(epoch_loss)), epoch_loss, label='Training Loss (MSE)')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Train Accuracy')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Final evaluation on test set
test_macro_auc = compute_macro_auc(model, X_test, y_test)
test_acc = accuracy(y_test, np.argmax(model.forward(X_test)[2], axis=1))
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Macro AUC: {test_macro_auc:.3f}")

##########################
### Keras MODEL fully connected
##########################

# Build the model
model_keras = keras.Sequential()  # Updated the name to model_keras
model_keras.add(keras.Input(shape=(28 * 28,)))
model_keras.add(Dense(units=500))
model_keras.add(Activation('sigmoid'))
model_keras.add(Dense(units=500))
model_keras.add(Activation('sigmoid'))
model_keras.add(Dense(units=10))
model_keras.add(Activation('softmax'))

# Compile the model
optimizer = SGD(learning_rate=0.1)
model_keras.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=10)
y_valid_onehot = to_categorical(y_valid, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)

# Train the model
history = model_keras.fit(X_train, y_train_onehot, validation_data=(X_valid, y_valid_onehot), batch_size=100, epochs=20)

# Evaluate the model
test_loss_keras, test_acc_keras = model_keras.evaluate(X_test, y_test_onehot)

# Calculate Macro AUC for Keras model
y_test_probas = model_keras.predict(X_test)
test_macro_auc_keras = roc_auc_score(y_test_onehot, y_test_probas, average="macro", multi_class="ovr")


# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Comparison between the Three Models:")

print("1. Single Hidden Layer:")
print(f"Test Accuracy: 94.54% ")
print("Test MSE: 0.3")


print("2. 2 Hidden Layers from Scratch:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Macro AUC: {test_macro_auc:.3f}")

print("3. Keras Fully Connected:")
print(f"Test Accuracy: {test_acc_keras * 100:.2f}%")
print(f"Test Macro AUC: {test_macro_auc_keras:.3f}")

