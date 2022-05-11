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
import pickle
from skimage.metrics import structural_similarity
#from image_similarity_measures.quality_metrics import rmse, ssim, sre
seed(1)
tf.compat.v1.set_random_seed(2)
import cv2


def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    rmse_bands = []
    for i in range(org_img.shape[2]):
        dif = np.subtract(org_img[:, :, i], pred_img[:, :, i])
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)



# with open('similarity_40_pos_and_neg/saved_best_matches_neg.pkl', 'rb') as f:
#     best_matches_dict = pickle.load(f)
# print(len(best_matches_dict[3]))
# print(best_matches_dict[3])
#Load data
# print("Loading data")
(X_train, Y_train) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_1")
(X_val, Y_val) = functions.read_data("data/DataSet_Nao_RAW/DataSet_SEQUENCE_2")
print(f'Train data shape: {X_train.shape}')
#
folder_name = 'similarity_40_pos_and_neg'
# # os.mkdir(f'{folder_name}')
# # functions.find_most_similar_pairs_from_class(X_train, Y_train, folder_name, top_similar=40)
#
#
# # FIRST model
# print("Generating pairs")
# (X_train_pairs, Y_train_pairs) = functions.generate_pairs_based_on_sim_pos_and_neg(X_train, Y_train, folder_name)
# (X_val_pairs, Y_val_pairs) = functions.generate_pairs(X_val, Y_val)
# print(f'Generated paris: {X_train_pairs.shape}')
# print(f'Generated paris val: {X_val_pairs.shape}')
#
# # siamese network
# imgA = Input(shape=(64, 64, 3))
# imgB = Input(shape=(64, 64, 3))
# featureExtractor = model.build_siamese_model()
# featsA = featureExtractor(imgA)  # embedding a
# featsB = featureExtractor(imgB)
#
# distance = Lambda(functions.euclidean_distance)([featsA, featsB])
# outputs = Dense(1, activation="sigmoid")(distance)
# model_1 = Model(inputs=[imgA, imgB], outputs=outputs)
# model_1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# # train
# his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
#                   validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=30)
#
#
#
# # save results
# os.mkdir(f'{folder_name}/model1')
# os.mkdir(f'{folder_name}/model1/plots')
# model_1.save_weights(f'{folder_name}/model1/model')
# functions.plot_learning_history_2(his, f'{folder_name}/model1')



# LOOP
print("Generating pairs")
for i in range(1, 40):
    (X_train_pairs, Y_train_pairs) = functions.generate_pairs_based_on_sim_pos_and_neg(X_train, Y_train, folder_name)
    (X_val_pairs, Y_val_pairs) = functions.generate_pairs(X_val, Y_val)
    print(f'Generated paris: {X_train_pairs.shape}')
    print(f'Generated paris val: {X_val_pairs.shape}')

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

    model_1.load_weights(f'similarity/model{i}/model')
    # train
    his = model_1.fit([X_train_pairs[:, 0], X_train_pairs[:, 1]], Y_train_pairs[:],
                      validation_data=([X_val_pairs[:, 0], X_val_pairs[:, 1]], Y_val_pairs[:]), batch_size=64, epochs=30)

    # save results
    os.mkdir(f'{folder_name}/model{i+1}')
    os.mkdir(f'{folder_name}/model{i+1}/plots')
    model_1.save_weights(f'{folder_name}/model{i+1}/model')
    functions.plot_learning_history_2(his, f'{folder_name}/model{i+1}')









####### finding similar
# i = 0
# lis = []
#
# cv2.imshow('image', X_train[2401])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# max = 100
# for x, y in zip(X_train, Y_train):
#
#     i += 1
#
#     if Y_train[2401] == y:
#         if np.array_equal(X_train[2401], x):
#             print('hi')
#             continue
#         else:
#             # print(f'{Y_train[2401]} vs {y}')
#             k = rmse(X_train[2401], x)
#             # print(k)
#             if k < max:
#                 max = k
#                 best_im = x
#             # cv2.imshow('image', x)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#             # lis.append(k)
#
# print('best_img')
# cv2.imshow('image', best_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#


#
# cv2.imshow('image', X_train[2800])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# sim_list = []
# sim_img = []
# for x, y in zip(X_train, Y_train):
#     if Y_train[2800] == y:
#         if np.array_equal(X_train[2800], x):
#             print('hi')
#             continue
#         else:
#             sim = rmse(X_train[2800], x)
#             sim_list.append(sim)
#             sim_img.append(x)
#
# def sort_index(lst, rev=True):
#     index = range(len(lst))
#     s = sorted(index, reverse=rev, key=lambda i: lst[i])
#     return s
#
# sorted_best = list(reversed(sort_index(sim_list)))
#
# top1_img = sim_img[sorted_best[0]]
# top2_img = sim_img[sorted_best[1]]
# top3_img = sim_img[sorted_best[2]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# top1_img = sim_img[sorted_best[5]]
# top2_img = sim_img[sorted_best[8]]
# top3_img = sim_img[sorted_best[20]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# top1_img = sim_img[sorted_best[25]]
# print('30')
# top2_img = sim_img[sorted_best[30]]
# top3_img = sim_img[sorted_best[40]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# cv2.imshow('image', X_train[2000])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# sim_list = []
# sim_img = []
# for x, y in zip(X_train, Y_train):
#     if Y_train[2000] == y:
#         if np.array_equal(X_train[2000], x):
#             print('hi')
#             continue
#         else:
#             sim = rmse(X_train[2000], x)
#             sim_list.append(sim)
#             sim_img.append(x)
#
# def sort_index(lst, rev=True):
#     index = range(len(lst))
#     s = sorted(index, reverse=rev, key=lambda i: lst[i])
#     return s
#
# sorted_best = list(reversed(sort_index(sim_list)))
#
# top1_img = sim_img[sorted_best[0]]
# top2_img = sim_img[sorted_best[1]]
# top3_img = sim_img[sorted_best[2]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# top1_img = sim_img[sorted_best[5]]
# top2_img = sim_img[sorted_best[8]]
# top3_img = sim_img[sorted_best[20]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# top1_img = sim_img[sorted_best[25]]
# print('30')
# top2_img = sim_img[sorted_best[30]]
# top3_img = sim_img[sorted_best[40]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
# cv2.imshow('image', X_train[6600])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# sim_list = []
# sim_img = []
# for x, y in zip(X_train, Y_train):
#     if Y_train[6600] == y:
#         if np.array_equal(X_train[6600], x):
#             print('hi')
#             continue
#         else:
#             sim = rmse(X_train[6600], x)
#             sim_list.append(sim)
#             sim_img.append(x)
#
# def sort_index(lst, rev=True):
#     index = range(len(lst))
#     s = sorted(index, reverse=rev, key=lambda i: lst[i])
#     return s
#
# sorted_best = list(reversed(sort_index(sim_list)))
#
# top1_img = sim_img[sorted_best[0]]
# top2_img = sim_img[sorted_best[1]]
# top3_img = sim_img[sorted_best[2]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# top1_img = sim_img[sorted_best[5]]
# top2_img = sim_img[sorted_best[8]]
# top3_img = sim_img[sorted_best[20]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top3_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# top1_img = sim_img[sorted_best[25]]
# print('30')
# top2_img = sim_img[sorted_best[30]]
# top3_img = sim_img[sorted_best[40]]
# print('best_img')
# cv2.imshow('image', top1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('best_img')
# cv2.imshow('image', top2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
