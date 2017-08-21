import sys

sys.path.append("D:/ML_Projects/BelgiumTS")
from data.belgiumTS import load_BelgiumTS
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers, optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(892)

VAL_PERCENTAGE = 0.3
IMAGE_SIZE = 28
NUM_CLASSES = 62
BATCH_SIZE = 60
LAYER_1_UNITS = 784
LAYER_2_UNITS = 196
LAYER_3_UNITS = 49
NB_EPOCHS = 10
LR_VAL = 0.001


def weight_initializer():
    return initializers.TruncatedNormal(stddev=0.1)


def bias_initializer():
    return initializers.Constant(value=0.1)


def mlp_model():
    model = Sequential()
    model.add(Dense(units=LAYER_1_UNITS, input_shape=(784,), use_bias=True,
                    activation='relu',
                    kernel_initializer=weight_initializer(),
                    bias_initializer=bias_initializer()))
    model.add(Dense(units=LAYER_2_UNITS, use_bias=True,
                    activation='relu',
                    kernel_initializer=weight_initializer(),
                    bias_initializer=bias_initializer()))
    model.add(Dense(units=LAYER_3_UNITS, use_bias=True,
                    activation='relu',
                    kernel_initializer=weight_initializer(),
                    bias_initializer=bias_initializer()))
    model.add(Dense(units=NUM_CLASSES, use_bias=True,
                    activation='softmax',
                    kernel_initializer=weight_initializer(),
                    bias_initializer=bias_initializer()))
    adam_opt = optimizers.Adam(lr=LR_VAL)

    model.compile(optimizer=adam_opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Training history for DL models
def plot_train_val_hist(train_perf, val_perf):
    # Plot cross entropy loss
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(train_perf[:,0], label="Training Loss")
    ax.plot(val_perf[:,0], label="Validation Loss")
    ax.set_title("Cross-Entropy Loss")
    plt.savefig("./keras_mlp/figures/Loss_Epochs.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(train_perf[:,1], label="Training Accuracy")
    ax.plot(val_perf[:,1], label="Validation Accuracy")
    ax.set_title("Accuracy")
    plt.savefig("./keras_mlp/figures/Accuracy_Epochs.png")

    return None


# Difference between model.fit() and model.train_on_batch() -
# model.fit() - breaks up data into small batches
# model.train_on_batch - uses the data it gets as a single batch, a single gradient update
# source - https://github.com/fchollet/keras/issues/68
def train(training_data, validation_data):
    model = mlp_model()
    val_perf, train_perf = [], []
    for i in range(NB_EPOCHS):
        val_perf_epoch = model.test_on_batch(x=validation_data.images, y=validation_data.labels)
        val_perf.append(val_perf_epoch)
        print("Validation Loss: %0.4f \t Validation Acc: %0.4f" % (val_perf_epoch[0], val_perf_epoch[1]))
        train_perf_batch = []
        for _ in range(training_data.num_examples // BATCH_SIZE):
            x, y = training_data.next_batch(BATCH_SIZE)
            train_perf_batch.append(model.train_on_batch(x=x, y=y))
        train_perf.append(np.mean(train_perf_batch, axis=0)) #Average loss and accuracy across batches
    plot_train_val_hist(np.array(train_perf), np.array(val_perf))
    return model


def test(model, testing_data):
    test_preds = model.predict_on_batch(testing_data.images)  # returns probabilities for each class
    test_preds = np.argmax(test_preds, axis=1)
    print("Test Acc: %0.4f" % (accuracy_score(y_true=np.argmax(testing_data.labels, axis=1), y_pred=test_preds)))
    return test_preds


def main(argv=None):
    belgiumTS_data = load_BelgiumTS(validation_size=VAL_PERCENTAGE, resize_pix=IMAGE_SIZE)

    training_data = belgiumTS_data.train
    validation_data = belgiumTS_data.validation
    testing_data = belgiumTS_data.test

    print("Training the model...")
    model = train(training_data, validation_data)

    print("Evaluate model...")
    test_preds = test(model, testing_data)


if __name__ == '__main__':
    main()
