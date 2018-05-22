import numpy as np
import matplotlib.pyplot as plt


def generate_linear_random(number, slope, bias):
    noise_scale = 0.1
    x_np = np.random.rand(number, 1)
    noise = np.random.normal(scale=noise_scale, size=(number, 1))
    y_np = x_np * slope + bias + noise
    return x_np, y_np


def generate_classes_random(number, number_classes):
    class_size = number // number_classes
    feature_shape = (class_size,)
    label_shape = (class_size, 1)
    x = []
    y = []
    for i in range(number_classes):
        mean = i * 2
        x_class = np.random.multivariate_normal(mean=np.array((mean, mean)), cov=.1 * np.eye(2), size=feature_shape)
        y_class = np.full(label_shape, i)
        x.append(x_class)
        y.append(y_class)

    x_np = np.vstack(x)
    y_np = np.vstack(y)

    return x_np, y_np

def one_hot(labels, number_of_classes):
    result = np.zeros((labels.shape[0], number_of_classes))
    for i, l in enumerate(labels):
        result[i, l] = 1
    return result


def generate_classes_random_multifeature(number, number_classes, number_features):
    class_size = number // number_classes
    feature_shape = (class_size, )
    label_shape = (class_size, 1)
    cov = .1 * np.eye(number_features)

    x = []
    y = []
    for i in range(number_classes):
        mean_array = np.full((number_features,), i * 2)
        x_class = np.random.multivariate_normal(mean=mean_array, cov=cov, size=feature_shape)
        y_class = np.full(label_shape, i)
        x.append(x_class)
        y.append(y_class)

    x_np = np.vstack(x)
    y_np = np.vstack(y)

    return x_np, one_hot(y_np, number_classes)


if __name__ == '__main__':
    # x, y = generate_linear_random(100, 5, 2)
    # plt.ion()
    # plt.scatter(x, y)
    # plt.show()

    plt.ion()
    x, y = generate_classes_random_multifeature(100, 4, 3)
    plt.scatter(x[:, 1], x[:, 2], c=y[:, 0])
    plt.show()
