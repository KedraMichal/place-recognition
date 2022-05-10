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
# seed(1)
# tf.compat.v1.set_random_seed(2)
from model import build_siamese_model

for i in range(32, 50):
    print(i)
    # Load data
    print("Loading data")
    (X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1")
    (X_val, Y_val) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2")
    print(f'Train data shape: {X_train.shape}')

    # Generate pairs
    print("Generating pairs")
    (X_train_pairs, Y_train_pairs) = functions.generate_pairs(X_train, Y_train)
    (X_val_pairs, Y_val_pairs) = functions.generate_pairs(X_val, Y_val)
    print(f'Generated paris: {X_train_pairs.shape}')


    # siamese network
    imgA = Input(shape=(64, 64, 3))
    imgB = Input(shape=(64, 64, 3))
    featureExtractor = model.build_siamese_model()
    featsA = featureExtractor(imgA)  # embedding a
    featsB = featureExtractor(imgB)

    distance = Lambda(functions.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model_1 = Model(inputs=[imgA, imgB], outputs=outputs)
    model_1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



    ## read saved model

    model_1.load_weights(f'result/modeltestowy_3_finetune_{i}/model')
    # train
    his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=25)

    # save results
    model_id = f'testowy_3_finetune_{i+1}'
    os.mkdir(f'result/model{model_id}')
    os.mkdir(f'result/model{model_id}/plots')
    model_1.save_weights(f'result/model{model_id}/model')
    functions.plot_learning_history(his, model_id)

