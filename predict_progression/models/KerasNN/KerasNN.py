from keras import Sequential, regularizers
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=100, kernel_initializer='uniform',
                    activation='softmax', kernel_constraint=maxnorm(4), kernel_regularizer=regularizers.l2(0.01),))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu',
                    kernel_constraint=maxnorm(4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad',
                  metrics=['accuracy'])
    return model


def get_model():
    return KerasClassifier(build_fn=baseline_model, epochs=2000,
                           batch_size=80, verbose=0, class_weight='balanced')
