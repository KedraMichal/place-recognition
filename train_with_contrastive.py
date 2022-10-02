import os
import functions
import model
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda


print("Loading data")
(X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1")
(X_val, Y_val) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2")
print(f'Train data shape: {X_train.shape}')
dir_name = 'contrastive'
os.mkdir(f'{dir_name}')

for i in range(1, 100):
    imgA = Input(shape=(64, 64, 3))
    imgB = Input(shape=(64, 64, 3))
    featureExtractor = model.build_siamese_model()
    featsA = featureExtractor(imgA)  # embedding a
    featsB = featureExtractor(imgB)
    distance = Lambda(functions.euclidean_distance)([featsA, featsB])

    model_1 = Model(inputs=[imgA, imgB], outputs=distance)
    model_1.compile(loss=functions.contrastive_loss, optimizer="adam", metrics=["accuracy"])
    if i != 1:
        model_1.load_weights(f'{dir_name}/model_with_cl_{i-1}/model')

    print("Generating pairs")
    (X_train_pairs, Y_train_pairs) = functions.generate_unique_pairs(X_train, Y_train,  dir_name)
    (X_val_pairs, Y_val_pairs) = functions.generate_pairs(X_val, Y_val)
    print(f'Generated paris: {X_train_pairs.shape}')
    his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                    validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=30)

    # save results
    os.mkdir(f'{dir_name}/model{i}')
    os.mkdir(f'{dir_name}/model{i}/plots')
    model_1.save_weights(f'{dir_name}/model{i}/model')
    functions.plot_learning_history(his, f'{dir_name}/model{i}/plots')

