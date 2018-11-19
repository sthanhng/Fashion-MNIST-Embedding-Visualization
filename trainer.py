import os

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

from utils import plot_loss_acc


def data_generator(batch_size=32):
    """
    Generates data containing batch size samples
    :param batch_size: The batch size
    :return:
    """

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    y_train = np_utils.to_categorical(y_train, 10)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    while True:
        for i in range(int(X_train.shape[0] / batch_size)):  # 1875
            if i % 125 == 0:
                print("| ==> [{0} / {1}]".format(i+1, int(X_train.shape[0] / batch_size)))
            # (32, 1, 28, 28), (32, 10)
            yield X_train[i * 32:(i + 1) * 32], y_train[i * 32:(i + 1) * 32]


def run_training(nb_epoch):
    """
    Training the Fashion MNIST model
    :param nb_epoch:
    :return:
    """

    # ===================================================================
    # Create the model
    # ===================================================================
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=[28, 28, 1]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # ===================================================================
    # Compile the model
    # ===================================================================
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])

    # ===================================================================
    # Train the model
    # ===================================================================
    print("[i] training the model...")
    history = model.fit_generator(data_generator(), steps_per_epoch=60000 // 32,
                        nb_epoch=nb_epoch, validation_data=None)

    # ===================================================================
    # Save the best model
    # ===================================================================
    if not os.path.exists("models/"):
        print("[i] creating the [models/] directory...")
        os.makedirs("models/")
    else:
        print("[i] the [models/] directory already exists!")

    print("[i] saving the model...")
    model.save_weights('./models/fashion_mnist_model.h5')
    json_string = model.to_json()

    with open('./models/config.json', 'w') as f:
        f.write(json_string)

    # ===================================================================
    # Plot the training loss and accuracy
    # ===================================================================
    plot_loss_acc(history, nb_epoch, "assets/training_loss_acc.png")

    print("********************* Done **********************")


if __name__ == "__main__":
    run_training(nb_epoch=10)