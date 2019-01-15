from sklearn.neural_network import MLPClassifier


def get_model():
    return MLPClassifier(hidden_layer_sizes=(256, 64), alpha=0.001,
                         solver='lbfgs', batch_size=80, verbose=0,
                         activation='relu', learning_rate='adaptive',
                         max_iter=5000)



