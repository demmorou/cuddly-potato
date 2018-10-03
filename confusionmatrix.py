import numpy as np
from sklearn.metrics import confusion_matrix
from model_cnn import model_cnn
from prepare_data import data_prepare


def confusion_matrixx(x_val, y_val, model):

    y_pred = np.argmax(model.predict(x_val, verbose=1), axis=1)
    matrix = confusion_matrix(y_val.astype(int), y_pred)

    print()
    print(matrix)


if __name__ == '__main__':
    model = model_cnn()
    model.load_weights('weights100.h5')

    x_train, y_train, x_val, y_val = data_prepare()

    confusion_matrixx(x_val, y_val, model)