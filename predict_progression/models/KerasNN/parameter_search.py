from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

from predict_progression.models.parameter_search import parameter_search


def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=296, kernel_initializer='uniform',
                    activation='softmax', kernel_constraint=maxnorm(4)))
    model.add(Dense(neurons, kernel_initializer='uniform', activation='relu',
                    kernel_constraint=maxnorm(4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad',
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=80,
                        epochs=2000)
neurons = [8, 16, 64, 128, 256, 512]
parameters = dict(neurons=neurons)
parameter_search(model=model, parameters=parameters, n_splits=10)
