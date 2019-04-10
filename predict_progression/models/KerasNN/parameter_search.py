from keras import Sequential, regularizers
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from predict_progression.models.parameter_search import parameter_search


def create_model(neurons=8):
    # create model
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(297,)))
    model.add(Dense(256, kernel_initializer='uniform',
                    activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=80,
                        epochs=500)
neurons = [8]
parameters = dict(neurons=neurons)
parameter_search(model=model, parameters=parameters, n_splits=10)
