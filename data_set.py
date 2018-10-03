import numpy as np
import urllib.request
import glob as g


def download():
    LABELS = np.array(['baseball', 'bowtie', 'clock', 'hand', 'hat',
                       'lightning', 'lollipop', 'mountain', 'pizza', 'potato', 'snowman', 'star'])

    for b in LABELS:
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(b)
        urllib.request.urlretrieve(url, "./data_set/{}.npy".format(b))


def load():
    return g.glob('./data_set/*.npy')


if __name__ == '__main__':
    download()