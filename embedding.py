import pickle
import random
import os

from keras.datasets import fashion_mnist
from keras.models import model_from_json, Model
from sklearn.decomposition import PCA


def run_embedding(mode='pca'):
    (X_train, y_train), (_, _) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    ids = random.sample(range(0, 60000), 10000)
    with open('./models/ids.pkl', 'wb') as f:
        pickle.dump(ids, f)

    X_train = X_train[ids]
    with open('./models/train_10k.pkl', 'wb') as f:
        pickle.dump(X_train, f)

    y_train = y_train[ids]
    if not os.path.exists('oss_data/'):
        print('[i] creating the [oss_data] directory...')
        os.makedirs('oss_data/')
    else:
        print('[i] the [oss_data/] directory already exists!')

    with open('./oss_data/Fashion_MNIST_labels.tsv', 'w') as f:
        for label in y_train:
            f.write(str(label) + '\n')

    with open('./models/config.json') as f:
        config = f.read()

    model = model_from_json(config)
    model.load_weights('./models/fashion_mnist_model.h5')
    new_model = Model(model.inputs, model.layers[-3].output)
    new_model.set_weights(model.get_weights())

    embedding_4096 = new_model.predict(X_train)
    if mode == 'pca':
        pca = PCA(n_components=128)
        embedding_128 = pca.fit_transform(embedding_4096)
        with open('./models/embedding_128D.pkl', 'wb') as f:
            pickle.dump(embedding_128, f)
        embedding_128.tofile('./oss_data/Fashion_MNIST_tensor.bytes')
    else:
        raise NotImplementedError('[!] The mode must be set!')


if __name__ == '__main__':
    run_embedding()
