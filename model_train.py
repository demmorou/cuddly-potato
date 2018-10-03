from model_cnn import model_cnn
from prepare_data import data_prepare
from confusionmatrix import confusion_matrixx
import keras


def model_train():

    model = model_cnn()

    x_train, y_train, x_val, y_val = data_prepare()

    y_train_onehot = keras.utils.to_categorical(y_train, 12)
    y_val_onehot = keras.utils.to_categorical(y_val, 12)

    print(x_train.shape)
    print(y_train_onehot.shape)
    print(x_val.shape)
    print(y_val_onehot.shape)

    print(model.summary())

    model.fit(x_train, y_train_onehot, batch_size=32, epochs=200, verbose=1)

    model.save_weights('weights200.h5')

    final_loss, final_acc = model.evaluate(x_val, y_val_onehot, verbose=1)
    print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

    confusion_matrixx(x_val, y_val, model)


if __name__ == '__main__':
    model_train()