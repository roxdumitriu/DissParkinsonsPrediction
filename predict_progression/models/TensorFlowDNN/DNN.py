import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd


def input_fn(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


data = pd.read_csv("../data/updrs.csv").drop(columns=["patno", "date_scan"])
y = data['score'].apply(int)
data = data.rename(columns=lambda x: x.replace("&", ""))
data["score"] = y
feature_columns = []
for feature in data.columns.values.tolist():
    if feature != "score":
        f = tf.feature_column.numeric_column(feature)
        feature_columns.append(f)

X_train, X_test, y_train, y_test = train_test_split(data, data["score"],
                                                    test_size=0.1)
train_data = lambda: input_fn(X_train, label_key='score', num_epochs=50,
                              shuffle=True, batch_size=10)
test_data = lambda: input_fn(X_test, label_key='score', num_epochs=50,
                             shuffle=True, batch_size=10)
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        n_classes=5, hidden_units=[512, 512, 512, 512, 512, 512],
                                        optimizer=tf.train.AdamOptimizer(
                                            # l1_regularization_strength=0.001,
                                            learning_rate=0.001
                                            ),
                                        )
classifier.train(train_data, steps=500)
result = classifier.evaluate(test_data)
print(result)
result = classifier.evaluate(train_data)
print(result)
# {'accuracy': 0.375, 'average_loss': 245.9207, 'loss': 2459.207, 'global_step': 36}
