from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


def default_model():
    """ Get the Keras model default parameters. Used to see the improvement of
        hyperparameter tuning.
     Returns
     ----------
     An MLP neural network with default parameters.
     """
    model = Sequential()
    model.add(Dense(8, input_dim=100, kernel_initializer='uniform',
                    activation="relu"))
    model.add(Dense(8, kernel_initializer='uniform',
                    activation="relu"))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    sgd = SGD(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def tuned_model():
    """ Get the Keras model with tuned parameters.
     Returns
     ----------
     An MLP neural network with optimal parameters.
     """
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(100,)))
    model.add(Dense(256, kernel_initializer='uniform',
                    activation="tanh",
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='uniform',
                    activation="relu"))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    sgd = SGD(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def get_model():
    """ Get the Sklearn wrapper of the Keras neural network. Used for
    interoperability with the Sklearn framework and the other models
    implemented in Sklearn.
    Returns
    ----------
    An Sklearn Keras wrapper of an MLP neural network with tuned parameters.
    """
    return KerasClassifier(
        build_fn=tuned_model,
        epochs=2000,
        batch_size=80, verbose=0, class_weight='balanced')


def get_default_model():
    """ Get the Sklearn wrapper of the Keras neural network. Used for
    interoperability with the Sklearn framework and the other models
    implemented in Sklearn.
    Returns
    ----------
    An Sklearn Keras wrapper of an MLP neural network with default parameters.
    """
    return KerasClassifier(build_fn=default_model, epochs=2000,
                           batch_size=80, verbose=0, class_weight="balanced")
