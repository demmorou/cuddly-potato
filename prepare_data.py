import numpy as np
import os
import matplotlib.pyplot as plt
img_rows, img_cols = 28, 28
data_dir = './data_set/'
ims_per_class = 2**12


def data_prepare():

    datas_path = filter(lambda x: x.endswith('.npy'), os.listdir(data_dir))
    dataset = np.array([]).reshape(0, img_rows * img_cols + 1)

    for i, d_path in enumerate(datas_path):
        data = np.load(os.path.join(data_dir, d_path))
        image_size = len(data)
        label = np.ones(image_size, dtype=int) * i
        data = np.concatenate((label[:, np.newaxis], data), axis=1)

        np.random.shuffle(data)

        dataset = np.append(dataset, data[0:ims_per_class], axis=0)

    np.random.shuffle(dataset)
    dataset_len = len(dataset)
    split_x = (int) (dataset_len * 0.9)

    print("Dataset {} images".format(dataset_len))
    print("Train {} images".format(split_x))
    print("Val {} images".format(dataset_len - split_x))

    train_data = dataset[0:split_x]
    val_data = dataset[split_x:-1]

    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    x_val = val_data[:, 1:]
    y_val = val_data[:, 0]

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    x_train /= 255
    x_val /= 255

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    data_prepare()