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
import triplet_functions


print("Loading data")
(X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1")
(X_val, Y_val) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2")
print(f'Train data shape: {X_train.shape}')
full_ind_train = triplet_functions.get_full_index(X_train, Y_train)
full_ind_test = triplet_functions.get_full_index(X_val, Y_val)
dir_name = 'triplets'
os.mkdir(f'{dir_name}')

for i in range(1, 101):
    # siamese network
    imgA = Input(shape=(64, 64, 3))
    imgB = Input(shape=(64, 64, 3))
    imgC = Input(shape=(64, 64, 3))
    featureExtractor = model.build_siamese_model()
    featsA = featureExtractor(imgA)  # embedding a
    featsB = featureExtractor(imgB)
    featsC = featureExtractor(imgC)
    loss = Lambda(triplet_functions.triplet_loss)([featsA, featsB, featsC])
    model_1 = Model(inputs=[imgA, imgB, imgC], outputs=loss)
    model_1.compile(loss=triplet_functions.identity_loss, optimizer="adam", metrics=["accuracy"])
    if i != 1:
        model_1.load_weights(f'{dir_name}/model_with_tl_{i-1}/model')

    X_train_triplets = triplet_functions.generate_triplets(X_train, Y_train, full_ind_train, 'triplets', save_json=True)
    X_val_triplets = triplet_functions.generate_triplets(X_val, Y_val, full_ind_test, 'triplets', save_json=True)
    batch = next(X_train_triplets)
    his = model_1.fit(X_train_triplets, validation_data=X_val_triplets, epochs=30, verbose=2, steps_per_epoch=20,
                    validation_steps=30)

    # save results
    os.mkdir(f'{dir_name}/model{i}')
    os.mkdir(f'{dir_name}/model{i}/plots')
    model_1.save_weights(f'{dir_name}/model{i}/model')
    functions.plot_learning_history(his, f'triplets/model{i}/plots')