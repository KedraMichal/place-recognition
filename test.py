import functions
from tensorflow.keras.models import load_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
import numpy as np
from keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


if __name__ == '__main__':
    SHAPE = (64, 64, 3)
    (X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1", shape=SHAPE)
    (X_test, Y_test) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_3", shape=SHAPE)

    imgA = Input(shape=SHAPE)
    imgB = Input(shape=SHAPE)
    imgC = Input(shape=SHAPE)

    featureExtractor = model.build_siamese_model()

    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    featsC = featureExtractor(imgC)

    # Binary CE
    distance = Lambda(functions.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    model.load_weights(f'result_unique_pairs/model50/model')
    # model.load_weights(f'deep_net_binary_ce/model50/model')
    # model.load_weights(f'deep_net_binary_ce2/model50/model')
    # model.load_weights(f'sim_photos_top_200/model4/model')

    # Contrastive loss
    # distance = Lambda(functions.euclidean_distance)([featsA, featsB])
    # model = Model(inputs=[imgA, imgB], outputs=distance)
    # model.load_weights(f'contrastive/model_with_cl_50/model')

    # Triplet loss
    # loss = Lambda(triplet_model.triplet_loss)([featsA, featsB, featsC])
    # model = Model(inputs=[imgA, imgB, imgC], outputs=loss)
    # model.load_weights(f'triplets/model_with_tl_100/model')

    X_train_extracted = featureExtractor.predict(X_train)
    X_test_extracted = featureExtractor.predict(X_test)

    for n in [1, 3, 5, 10, 20]:
        print(f'KNN - {n}')
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train_extracted, Y_train)
        y_pred = knn.predict(X_test_extracted)
        print("Accuracy: ", accuracy_score(Y_test, y_pred))
        print("F1: ", f1_score(Y_test, y_pred, average='weighted'))
        print("Precision: ", precision_score(Y_test, y_pred, average='weighted'))
        print("Recall: ", recall_score(Y_test, y_pred, average='weighted'))
        print(confusion_matrix(Y_test, y_pred))

    print('SVM')
    svm = SVC().fit(X_train_extracted, Y_train)
    y_pred = svm.predict(X_test_extracted)
    print("Accuracy: ", accuracy_score(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))
    print("F1: ", f1_score(Y_test, y_pred, average='weighted'))
    print("Precision: ", precision_score(Y_test, y_pred, average='weighted'))
    print("Recall: ", recall_score(Y_test, y_pred, average='weighted'))
    cm = confusion_matrix(Y_test, y_pred)

    # Confusion matrix
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={"size": 19}) # annot=True to annotate cells, ftm='g' to disable scientific notation
    # ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=19)
    # ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=19)
    # # labels, title and ticks
    # ax.set_xlabel('Klasa predykowana', fontsize=29)
    # ax.set_ylabel('Klasa rzeczywista', fontsize=29)
    # ax.set_title('Macierz błedów', fontsize=40)
    # plt.show()


