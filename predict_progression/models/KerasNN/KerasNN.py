from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


def baseline_model():
    # create model
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
    return KerasClassifier(
        build_fn=baseline_model,
        epochs=2000,
        batch_size=80, verbose=0, class_weight='balanced')


def get_default_model():
    return KerasClassifier(build_fn=baseline_model, epochs=2000,
                           batch_size=80, verbose=0, class_weight="balanced")
