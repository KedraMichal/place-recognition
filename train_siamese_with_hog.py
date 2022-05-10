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


SHAPE_HOG=(120, 120, 1)

# Load data
print("Loading data")
(X_train, Y_train) = functions.read_data_with_hog("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1", shape=SHAPE_HOG)
print(X_train.shape)
(X_val, Y_val) = functions.read_data_with_hog("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2", shape=SHAPE_HOG)
print(f'Train data shape: {X_train.shape}')
print(f'Train val shape: {X_val.shape}')


# for i in range(38, 70):
# Generate pairs
print("Generating pairs")
(X_train_pairs, Y_train_pairs) = functions.generate_pairs(X_train, Y_train)
(X_val_pairs, Y_val_pairs) = functions.generate_pairs(X_val, Y_val)
print(f'Generated paris: {X_train_pairs.shape}')


# siamese network
imgA = Input(shape=SHAPE_HOG)
imgB = Input(shape=SHAPE_HOG)
featureExtractor = model.build_siamese_model_hog(shape=SHAPE_HOG)
featsA = featureExtractor(imgA)  # embedding a
featsB = featureExtractor(imgB)

distance = Lambda(functions.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model_1 = Model(inputs=[imgA, imgB], outputs=outputs)
model_1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model_1.load_weights(f'hog/hog')
# train
his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=100)

# save results

model_id = f'hog_2'
os.mkdir(f'{model_id}')
os.mkdir(f'{model_id}/plots')
model_1.save_weights(f'{model_id}/hog_2')
functions.plot_learning_history_2(his, path=f'{model_id}/plots')


his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=100)

# save results

model_id = f'hog_3'
os.mkdir(f'{model_id}')
os.mkdir(f'{model_id}/plots')
model_1.save_weights(f'{model_id}/hog_3')
functions.plot_learning_history_2(his, path=f'{model_id}/plots')


his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=100)

# save results

model_id = f'hog_4'
os.mkdir(f'{model_id}')
os.mkdir(f'{model_id}/plots')
model_1.save_weights(f'{model_id}/hog_4')
functions.plot_learning_history_2(his, path=f'{model_id}/plots')

his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=100)

# save results

model_id = f'hog_5'
os.mkdir(f'{model_id}')
os.mkdir(f'{model_id}/plots')
model_1.save_weights(f'{model_id}/hog_5')
functions.plot_learning_history_2(his, path=f'{model_id}/plots')


his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=100)

# save results

model_id = f'hog_6'
os.mkdir(f'{model_id}')
os.mkdir(f'{model_id}/plots')
model_1.save_weights(f'{model_id}/hog_6')
functions.plot_learning_history_2(his, path=f'{model_id}/plots')

