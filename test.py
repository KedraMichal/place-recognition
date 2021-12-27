import functions
from tensorflow.keras.models import load_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
import numpy as np


def predict_class(x_test, x_query, y_query):
    predicted_labels, probabilities = [], []
    for ind, x in enumerate(x_test):
        #print(ind)
        x = [x] * x_query.shape[0]
        x = np.array(x)
        predictions_prob = model.predict([x, x_query])
        probabilities.append(predictions_prob)
        predicted_labels.append(max(list(zip(predictions_prob, y_query)), key=lambda ind: ind[0])[1])

    return predicted_labels, probabilities


def calculate_mAP(y_test, y_query, probabilities, pred_labels):
    average_precision_scores = []
    for label in np.unique(y_test):
        all_x_prob = []
        for prob_ind in range(len(probabilities)):
            prob_of_x = []
            for prob, y in zip(probabilities[prob_ind], y_query):
                if y == label:
                    prob_of_x.append(prob)
            all_x_prob.append(max(prob_of_x))
        y_true = []
        for y in pred_labels:
            if y == label:
                y_true.append(1)
            else:
                y_true.append(0)
        average_precision_scores.append(average_precision_score(y_true, all_x_prob))
    return np.mean(np.array(average_precision_scores))


if __name__ == '__main__':
    (X_test, Y_test) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2")
    (X_query, Y_query) = functions.read_data("data/DataSet_Nao_PlaceRecognition/SEQUENCE_2", True)

    model = load_model("result/model_30e3")
    pred_labels, probabilities = predict_class(X_test, X_query, Y_query)
    mAP = calculate_mAP(Y_test, Y_query, probabilities, pred_labels)
    print("mAP:", mAP)
    print("Accuracy: ", accuracy_score(Y_test, pred_labels))
    print(confusion_matrix(Y_test, pred_labels))


