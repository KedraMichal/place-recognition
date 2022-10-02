import functions
from tensorflow.keras.models import load_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import model
import triplet_functions
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import rcParams


def plot_2d_scatter(X, y):
    fig, plot = plt.subplots()
    # fig.set_size_inches(16, 16)
    plt.prism()
    y = np.array(y).astype(int)
    for i in range(9):
        digit_indices = y==i
        plot.scatter(X[digit_indices, 0], X[digit_indices, 1], label=i)

    plot.set_xticks(())
    plot.set_yticks(())
    plt.tight_layout()
    plt.legend(title="Klasa", fontsize=22, title_fontsize=30, loc="upper right")


rcParams.update({'figure.autolayout': True})
SHAPE = (64, 64, 3)
(X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1", shape=SHAPE)
(X_test, Y_test) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2", shape=SHAPE)

imgA = Input(shape=SHAPE)
imgB = Input(shape=SHAPE)
imgC = Input(shape=SHAPE)
featureExtractor = model.build_siamese_model()

featsA = featureExtractor(imgA)  # embedding a
featsB = featureExtractor(imgB)
featsC = featureExtractor(imgC)

# Binary CE
distance = Lambda(functions.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
model.load_weights(f'deep_net_binary_ce/model50/model')

X_train_extracted = featureExtractor.predict(X_train)
X_test_extracted = featureExtractor.predict(X_test)

X_small = X_train_extracted[:]
y_small = Y_train[:]

tsne = TSNE()
X_tsne_embedded = tsne.fit_transform(X_small)
print(np.unique(np.asarray(y_small)))
plot_2d_scatter(X_tsne_embedded, y_small)
plt.title("Wynik t-SNE uzyskane z wykorzystaniem sieci syjamskiej jako ekstraktora cech").set_fontsize('40')
plt.xlabel('Wymiar 1').set_fontsize('32')
plt.ylabel('Wymiar 2').set_fontsize('32')
plt.show()

