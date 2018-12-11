import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

N_SPLITS = 9


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=4, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def process_data(df):
    X = df.drop(columns=["patno", "score", "date_scan"])
    y = df["score"].astype(int)
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    return X, y


avg_accuracy = 0
for i in range(0, N_SPLITS):
    data_test = pd.read_csv("../data/updrs_splits/split_{}.csv".format(i))
    data_train = pd.DataFrame()
    for j in range(0, N_SPLITS):
        if i != j:
            split = pd.read_csv("../data/updrs_splits/split_{}.csv".format(j))
            data_train = pd.concat([data_train, split])
    X_train, y_train = process_data(data_train)
    X_test, y_test = process_data(data_test)

    classifier = KerasClassifier(build_fn=baseline_model, epochs=200,
                                 batch_size=5, verbose=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    avg_accuracy += accuracy
    print("Split {0} accuracy: {1}".format(i, accuracy))

print()
print("Average accuracy: {}".format(avg_accuracy / N_SPLITS))
