import numpy as np
import pickle
import scipy.misc


def images_to_sprite(data):
    """
    Creates the sprite image
    :param data: [batch_size, height, weight, n_channel]
    :return data: The sprited image::[height, weight, n_channel]
    """

    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)

    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)

    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)

    return data


if __name__ == "__main__":
    with open('./models/train_10k.pkl', 'rb') as f:
        X_train = pickle.load(f)

    sprite_img = images_to_sprite(X_train)
    scipy.misc.imsave('./oss_data/Fashion_MNIST_sprites.png', np.squeeze(sprite_img))
