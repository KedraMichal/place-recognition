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
import cv2

# def predict_class(x_test, x_query, y_query):
#     predicted_labels, probabilities = [], []
#     for ind, x in enumerate(x_test):
#         #print(ind)
#         x = [x] * x_query.shape[0]
#         x = np.array(x)
#         predictions_prob = model.predict([x, x_query])
#         probabilities.append(predictions_prob)
#         predicted_labels.append(max(list(zip(predictions_prob, y_query)), key=lambda ind: ind[0])[1])
#
#     return predicted_labels, probabilities


# def calculate_mAP(y_test, y_query, probabilities, pred_labels):
#     average_precision_scores = []
#     for label in np.unique(y_test):
#         all_x_prob = []
#         for prob_ind in range(len(probabilities)):
#             prob_of_x = []
#             for prob, y in zip(probabilities[prob_ind], y_query):
#                 if y == label:
#                     prob_of_x.append(prob)
#             all_x_prob.append(max(prob_of_x))
#         y_true = []
#         for y in pred_labels:
#             if y == label:
#                 y_true.append(1)
#             else:
#                 y_true.append(0)
#         average_precision_scores.append(average_precision_score(y_true, all_x_prob))
#     return np.mean(np.array(average_precision_scores))


if __name__ == '__main__':
    #SHAPE = (128, 128, 3)
    SHAPE = (64, 64, 3)
    (X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1", shape=SHAPE)
    (X_test, Y_test) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2", shape=SHAPE)
    print(X_train.shape)
    #SHAPE = (64, 64, 3)
    #SHAPE = (160, 160, 1)

    # (X_train, Y_train) = functions.read_data_with_hog("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1", shape=SHAPE)
    # (X_test, Y_test) = functions.read_data_with_hog("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2", shape=SHAPE)



    imgA = Input(shape=SHAPE)
    imgB = Input(shape=SHAPE)
    featureExtractor = model.build_siamese_model()
    #featureExtractor = model.build_siamese_model(shape=SHAPE)
    #featureExtractor = model.build_siamese_model_hog(SHAPE)
    featsA = featureExtractor(imgA)  # embedding a
    featsB = featureExtractor(imgB)

    distance = Lambda(functions.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)
    model.load_weights(f'similarity/model8/model')
    #model.load_weights(f'result_unique_pairs/modeltestowy_15/model')
    #model.load_weights(f'hog/hog')
    #model.load_weights(f'result_unique_pairs_deeper/modeltestowy_24/model')

    # pred_labels, probabilities = predict_class(X_test, X_query, Y_query)
    # mAP = calculate_mAP(Y_test, Y_query, probabilities, pred_labels)
    # # print("mAP:", mAP)
    # # print("Accuracy: ", accuracy_score(Y_test, pred_labels))
    # # print(confusion_matrix(Y_test, pred_labels))
    from keras import backend as K

    # with a Sequential model
    # get_3rd_layer_output = K.function([model.layers[0].input],
    #                                   [model.layers[3].output])
    # layer_output = get_3rd_layer_output([x])[0]


    X_train_extracted = featureExtractor.predict(X_train)
    X_test_extracted = featureExtractor.predict(X_test)
    # print(X_train_extracted.shape)
    # print(X_test_extracted.shape)
    # print(Y_train.shape)
    # print(Y_test.shape)
    # print(Y_train)
    for n in [3, 5, 7, 8, 9, 12, 15]:
        print(f'KNN - {n}')
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train_extracted, Y_train)
        y_pred = knn.predict(X_test_extracted)
        print("Accuracy: ", accuracy_score(Y_test, y_pred))
        # print("F1: ", f1_score(Y_test, y_pred, average='weighted'))
        # print("Precision: ", precision_score(Y_test, y_pred, average='weighted'))
        # print("Recall: ", recall_score(Y_test, y_pred, average='weighted'))
        print(confusion_matrix(Y_test, y_pred))

    print('SVM')
    svm = SVC().fit(X_train_extracted, Y_train)
    y_pred = svm.predict(X_test_extracted)
    print("Accuracy: ", accuracy_score(Y_test, y_pred))

    print(confusion_matrix(Y_test, y_pred))
    # print(Y_test)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_train_extracted.shape)
    # for ind, (act_class, pred_class) in enumerate(zip(Y_test, y_pred)):
    #     if act_class != pred_class:
    #         print(ind)
    #         print(act_class)
    #         print(pred_class)
    #         print(X_test[ind].shape)
    #         cv2.imshow('image', X_test[ind])
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
            # raise ValueError
    # print("F1: ", f1_score(Y_test, y_pred,average='weighted'))
    # print("Precision: ", precision_score(Y_test, y_pred,average='weighted'))
    # print("Recall: ", recall_score(Y_test, y_pred,average='weighted'))
