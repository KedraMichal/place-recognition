import os

import functions
import model
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from numpy.random import seed
import tensorflow as tf
seed(1)
tf.compat.v1.set_random_seed(2)


SHAPE = (128, 128, 3)

# Load data
print("Loading data")
(X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1", shape=SHAPE)
print(X_train.shape)
(X_val, Y_val) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2", shape=SHAPE)
print(f'Train data shape: {X_train.shape}')
print(f'Train val shape: {X_val.shape}')


# for i in range(38, 70):
# Generate pairs

for i in range(27, 50):
    print("Generating pairs")
    (X_train_pairs, Y_train_pairs) = functions.generate_unique_pairs_2(X_train, Y_train)
    (X_val_pairs, Y_val_pairs) = functions.generate_unique_pairs_2(X_val, Y_val)
    print(f'Generated paris: {X_train_pairs.shape}')
    print(f'Generated paris val: {X_val_pairs.shape}')


    # siamese network
    imgA = Input(shape=SHAPE)
    imgB = Input(shape=SHAPE)
    featureExtractor = model.build_siamese_model_2(shape=SHAPE)
    featsA = featureExtractor(imgA)  # embedding a
    featsB = featureExtractor(imgB)

    distance = Lambda(functions.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model_1 = Model(inputs=[imgA, imgB], outputs=outputs)
    model_1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    if os.path.exists(f'result_unique_pairs_deeper/modeltestowy_{i}/plots'):
        print('hi')
        model_1.load_weights(f'result_unique_pairs_deeper/modeltestowy_{i}/model')
    # train
    his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                    validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=30)

    # save results
    model_id = f'testowy_{i+1}'
    os.mkdir(f'result_unique_pairs_deeper/model{model_id}')
    os.mkdir(f'result_unique_pairs_deeper/model{model_id}/plots')
    model_1.save_weights(f'result_unique_pairs_deeper/model{model_id}/model')
    functions.plot_learning_history_2(his, f'result_unique_pairs_deeper/model{model_id}/plots')

    del X_train_pairs
    del Y_train_pairs
    del X_val_pairs
    del Y_val_pairs

