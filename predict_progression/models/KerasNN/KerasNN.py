from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


def baseline_model(dropout_layer_1=0.0, dropout_layer_2=0.0, neurons_layer_1=8,
                   neurons_layer_2=8, activation_layer_1="relu",
                   activation_layer_2="relu", optimizer="Adagrad"):
    # create model
    model = Sequential()
    model.add(Dropout(dropout_layer_1, input_shape=(100,)))
    model.add(Dense(neurons_layer_1, kernel_initializer='uniform',
                    activation=activation_layer_1,
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_layer_2))
    model.add(Dense(neurons_layer_2, kernel_initializer='uniform',
                    activation=activation_layer_2))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    # sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def get_model():
    return KerasClassifier(
        build_fn=baseline_model(dropout_layer_1=0.5, dropout_layer_2=0.5,
                                neurons_layer_1=256, neurons_layer_2=64,
                                activation_layer_1="tanh",
                                activation_layer_2="relu", optimizer="Adagrad"),
        epochs=2000,
        batch_size=80, verbose=0, class_weight='balanced')


def get_default_model():
    return KerasClassifier(build_fn=baseline_model, epochs=2000,
                           batch_size=80, verbose=0, class_weight="balanced")
