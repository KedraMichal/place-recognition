import glob
import cv2
import numpy as np
#import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import json
import pickle
from skimage import data, exposure
from skimage.feature import hog



def read_data(path, queries=False, shape=(64, 64, 1)):
    images, string_labels = [], []
    if queries is False:
        where_images = glob.glob('{}/**/*.png'.format(path))
    else:
        where_images = glob.glob("{}/**/query/*.png".format(path), recursive=True)

    for path in where_images:
        path_split = path.split('\\')
        image_name = path_split[-1]
        label = image_name.split('_')[0]

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # rgb (1)
        #print(img.shape)   #480 na 640 pixeli
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = cv2.resize(img, (shape[0], shape[1]))
        #print(img.shape) # (96,72) -> 72x96
        images.append(np.asarray(img))
        string_labels.append(np.asarray(label))
    unique_labels = np.unique(string_labels)
    numeric_labels = []

    for label in string_labels:
        for ind, label2 in enumerate(unique_labels):
            if label == label2:
                numeric_labels.append(ind)
                break

    return np.asarray(images) / 255.0, np.asarray(numeric_labels)


def read_data_with_hog(path, queries=False, shape=(360, 360, 1)):
    images, string_labels = [], []
    if queries is False:
        where_images = glob.glob('{}/**/*.png'.format(path))
    else:
        where_images = glob.glob("{}/**/query/*.png".format(path), recursive=True)
    a = 0
    for path in where_images:
        a += 1
        if a % 1 == 0:
            path_split = path.split('\\')
            image_name = path_split[-1]
            label = image_name.split('_')[0]

            img = cv2.imread(path, cv2.IMREAD_COLOR)  # rgb (1)
            img = cv2.resize(img, (shape[0], shape[1]))
            fd, hog_image = hog(img, visualize=True, channel_axis=-1)
            hog_image = hog_image.reshape(*hog_image.shape, 1)
            print(hog_image.shape)
            print(a)
            images.append(np.asarray(hog_image))
            string_labels.append(np.asarray(label))
    unique_labels = np.unique(string_labels)
    numeric_labels = []

    for label in string_labels:
        for ind, label2 in enumerate(unique_labels):
            if label == label2:
                numeric_labels.append(ind)
                break

    return np.asarray(images) / 255.0, np.asarray(numeric_labels)


def generate_pairs(images, labels):
    unique_labels = np.unique(labels)

    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    pair_images, pair_labels = [], []
    i = 0
    for image, label in zip(images, labels):

        positive_images = full_index_list[label]
        random_positive_image_index = np.random.choice(positive_images)
        random_positive_image = images[random_positive_image_index]

        pair_images.append([image, random_positive_image])
        pair_labels.append([1])


        negative_images = full_index_list[:label] + full_index_list[label + 1:]
        negative_images_flatten = [item for sublist in negative_images for item in sublist]

        random_negative_image_index = np.random.choice(negative_images_flatten)
        random_negative_image = images[random_negative_image_index]

        pair_images.append([image, random_negative_image])
        pair_labels.append([0])

        i = i+1

    return np.asarray(pair_images), np.asarray(pair_labels)


def generate_unique_pairs(images, labels):
    unique_labels = np.unique(labels)

    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    pair_images, pair_labels = [], []

    try:
        with open('result_unique_pairs/pos_pairs.json', 'rb') as f:
            #pos_pair_ind = [tuple(x) for x in json.load(f)]
            pos_pair_ind = pickle.load(f)
        with open('result_unique_pairs/neg_pairs.json', 'rb') as f:
            #neg_pair_ind = [tuple(x) for x in json.load(f)]
            neg_pair_ind = pickle.load(f)
    except:
        pos_pair_ind, neg_pair_ind = [], []

    i = 0
    for _, label in zip(images, labels):
        print(i)
        if i % 3 == 0:
            print(i)
            positive_images = full_index_list[label]

            while True:
                random_positive_image_indexes = np.random.choice(positive_images, 2, replace=False)
                first_elem, sec_elem = random_positive_image_indexes[0], random_positive_image_indexes[1]
                tuple1, tuple2 = (first_elem, sec_elem), (sec_elem, first_elem)
                if tuple1 not in pos_pair_ind:
                    if tuple2 not in pos_pair_ind:
                        break
            random_positive_image_index_1 = random_positive_image_indexes[0]
            random_positive_image_index_2 = random_positive_image_indexes[1]
            random_positive_image_1 = images[random_positive_image_index_1]
            random_positive_image_2 = images[random_positive_image_index_2]
            pair_images.append([random_positive_image_1, random_positive_image_2])
            pair_labels.append([1])


            ##########
            negative_images = full_index_list[:label] + full_index_list[label + 1:]
            negative_images_flatten = [item for sublist in negative_images for item in sublist]
            while True:
                random_negative_image_index = np.random.choice(negative_images_flatten)
                tuple1, tuple2 = (first_elem, random_negative_image_index), (random_negative_image_index, first_elem)
                if tuple1 not in pos_pair_ind:
                    if tuple2 not in pos_pair_ind:
                        break
            random_negative_image = images[random_negative_image_index]
            pair_images.append([random_positive_image_1, random_negative_image])
            pair_labels.append([0])

            pos_pair_ind.append((random_positive_image_index_1, random_positive_image_index_2))
            neg_pair_ind.append((random_positive_image_index_1, random_negative_image_index))
        i += 1
    with open('result_unique_pairs/neg_pairs.json', 'wb') as f:
        pickle.dump(neg_pair_ind, f)
    with open('result_unique_pairs/pos_pairs.json', 'wb') as f:
        pickle.dump(pos_pair_ind, f)

    return np.asarray(pair_images), np.asarray(pair_labels)


def generate_unique_pairs_2(images, labels):
    unique_labels = np.unique(labels)

    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    pair_images, pair_labels = [], []

    try:
        with open('result_unique_pairs_deeper/pos_pairs.json', 'rb') as f:
            #pos_pair_ind = [tuple(x) for x in json.load(f)]
            pos_pair_ind = pickle.load(f)
        with open('result_unique_pairs_deeper/neg_pairs.json', 'rb') as f:
            #neg_pair_ind = [tuple(x) for x in json.load(f)]
            neg_pair_ind = pickle.load(f)
    except:
        pos_pair_ind, neg_pair_ind = [], []

    i = 0
    for _, label in zip(images, labels):
        positive_images = full_index_list[label]

        while True:
            random_positive_image_indexes = np.random.choice(positive_images, 2, replace=False)
            first_elem, sec_elem = random_positive_image_indexes[0], random_positive_image_indexes[1]
            tuple1, tuple2 = (first_elem, sec_elem), (sec_elem, first_elem)
            if tuple1 not in pos_pair_ind:
                if tuple2 not in pos_pair_ind:
                    break
        random_positive_image_index_1 = random_positive_image_indexes[0]
        random_positive_image_index_2 = random_positive_image_indexes[1]
        random_positive_image_1 = images[random_positive_image_index_1]
        random_positive_image_2 = images[random_positive_image_index_2]


        ##########
        negative_images = full_index_list[:label] + full_index_list[label + 1:]
        negative_images_flatten = [item for sublist in negative_images for item in sublist]
        while True:
            random_negative_image_index = np.random.choice(negative_images_flatten)
            tuple1, tuple2 = (first_elem, random_negative_image_index), (random_negative_image_index, first_elem)
            if tuple1 not in pos_pair_ind:
                if tuple2 not in pos_pair_ind:
                    break
        random_negative_image = images[random_negative_image_index]

        if i % 3 == 0:
            pair_images.append([random_positive_image_1, random_positive_image_2])
            pair_labels.append([1])
            pos_pair_ind.append((random_positive_image_index_1, random_positive_image_index_2))
            pair_images.append([random_positive_image_1, random_negative_image])
            pair_labels.append([0])
            neg_pair_ind.append((random_positive_image_index_1, random_negative_image_index))

        i += 1
    with open('result_unique_pairs_deeper/neg_pairs.json', 'wb') as f:
        pickle.dump(neg_pair_ind, f)
    with open('result_unique_pairs_deeper/pos_pairs.json', 'wb') as f:
        pickle.dump(pos_pair_ind, f)

    return np.asarray(pair_images), np.asarray(pair_labels)




# def generate_pairs_based_on_similarity(images, labels, siamese_net):
#     unique_labels = np.unique(labels)
#
#     full_index_list = []
#     for ind_lab in unique_labels:
#         ind_list = []
#         for ind, lab in enumerate(labels):
#             if lab == ind_lab:
#                 ind_list.append(ind)
#         full_index_list.append(ind_list)
#
#     pair_images, pair_labels = [], []
#     i = 0
#     for image, label in zip(images, labels):
#
#         positive_images = full_index_list[label]
#
#         samples = 0
#         while True:
#             random_positive_image_index = np.random.choice(positive_images)
#             random_positive_image = images[random_positive_image_index]
#             image0 = np.expand_dims(image, axis=0)
#             image1 = np.expand_dims(random_positive_image, axis=0)
#             samples += 1
#             if siamese_net.predict([image0, image1]) > 0.995:
#                 pair_images.append([image, random_positive_image])
#                 pair_labels.append([1])
#             if samples == 100:
#                 samples = 0
#                 break
#
#
#         negative_images = full_index_list[:label] + full_index_list[label + 1:]
#         negative_images_flatten = [item for sublist in negative_images for item in sublist]
#         while True:
#             random_negative_image_index = np.random.choice(negative_images_flatten)
#             random_negative_image = images[random_negative_image_index]
#             image2 = np.expand_dims(random_negative_image, axis=0)
#             samples += 1
#             if siamese_net.predict([image0, image2]) > 0.995:
#                 pair_images.append([image, random_positive_image])
#                 pair_labels.append([1])
#             if samples == 100:
#                 samples = 0
#                 break
#         pair_images.append([image, random_negative_image])
#         pair_labels.append([0])
#
#         i = i+1
#
#     return np.asarray(pair_images), np.asarray(pair_labels)


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


def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s


def find_most_similar_pairs_from_class(images, labels, path, top_similar=30):
    unique_labels = np.unique(labels)
    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    best_matches_dict = {}
    best_matches_neg_dict = {}
    kk = 0
    for image, label in zip(images, labels):
        print(kk)
        kk += 1
        positive_images = full_index_list[label]
        sim_list = []
        positive_images_ind_list = []
        for pos_img in positive_images: # example : [1600, 1601, 1602 ... 2004]
            positive_image = images[pos_img]
            if np.array_equal(positive_image, image):
                base_img_index = pos_img
                continue
            sim = rmse(image, positive_image)
            sim_list.append(sim)
            positive_images_ind_list.append(pos_img)

        sorted_best = list(reversed(sort_index(sim_list)))
        best_matches = []
        for i, ind in enumerate(sorted_best):
            best_matches.append(positive_images_ind_list[ind])
            if i == top_similar:
                break
        best_matches_dict[base_img_index] = best_matches


        negative_images = full_index_list[:label] + full_index_list[label + 1:]
        negative_images_flatten = [item for sublist in negative_images for item in sublist]
        sim_neg_list = []
        negative_images_ind_list = []
        for neg_img in negative_images_flatten: # example : [1600, 1601, 1602 ... 2004]
            negative_image = images[neg_img]
            sim = rmse(image, negative_image)
            sim_neg_list.append(sim)
            negative_images_ind_list.append(neg_img)

        sorted_best_neg = list(reversed(sort_index(sim_neg_list)))
        best_matches_neg = []
        for i, ind in enumerate(sorted_best_neg):
            best_matches_neg.append(negative_images_ind_list[ind])
            if i == top_similar:
                break
        best_matches_neg_dict[base_img_index] = best_matches_neg

    with open(f'{path}/saved_best_matches.pkl', 'wb') as f:
        pickle.dump(best_matches_dict, f)

    with open(f'{path}/saved_best_matches_neg.pkl', 'wb') as f:
        pickle.dump(best_matches_neg_dict, f)

    return None


def generate_pairs_based_on_sim(images, labels, path):
    unique_labels = np.unique(labels)

    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    with open(f'{path}/saved_best_matches.pkl', 'rb') as f:
        best_matches_dict = pickle.load(f)
    pair_images, pair_labels = [], []

    base_img_ind = 0
    for image, label in zip(images, labels):
        positive_images = full_index_list[label]
        # print('best_img')
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        random_positive_image_index = np.random.choice(best_matches_dict[base_img_ind])
        random_positive_image = images[random_positive_image_index]

        # print('best_img')
        # cv2.imshow('image', random_positive_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pair_images.append([image, random_positive_image])
        pair_labels.append([1])

        negative_images = full_index_list[:label] + full_index_list[label + 1:]
        negative_images_flatten = [item for sublist in negative_images for item in sublist]
        random_negative_image_index = np.random.choice(negative_images_flatten)
        random_negative_image = images[random_negative_image_index]

        pair_images.append([image, random_negative_image])
        pair_labels.append([0])

        base_img_ind += 1

    return np.asarray(pair_images), np.asarray(pair_labels)



def generate_pairs_based_on_sim_pos_and_neg(images, labels, path):
    unique_labels = np.unique(labels)

    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    with open(f'{path}/saved_best_matches.pkl', 'rb') as f:
        best_matches_dict = pickle.load(f)
    with open(f'{path}/saved_best_matches_neg.pkl', 'rb') as f:
        best_matches_neg_dict = pickle.load(f)

    pair_images, pair_labels = [], []

    base_img_ind = 0
    for image, label in zip(images, labels):
        #positive_images = full_index_list[label]
        # print('base')
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        random_positive_image_index = np.random.choice(best_matches_dict[base_img_ind])
        random_positive_image = images[random_positive_image_index]

        old_list = best_matches_dict[base_img_ind]
        old_list.remove(random_positive_image_index)
        best_matches_dict[base_img_ind] = old_list

        # print('pos')
        # cv2.imshow('image', random_positive_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pair_images.append([image, random_positive_image])
        pair_labels.append([1])

        #negative_images = full_index_list[:label] + full_index_list[label + 1:]
        #negative_images_flatten = [item for sublist in negative_images for item in sublist]
        random_negative_image_index = np.random.choice(best_matches_neg_dict[base_img_ind])
        random_negative_image = images[random_negative_image_index]
        # print('neg')
        # cv2.imshow('image', random_negative_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        old_list_neg = best_matches_neg_dict[base_img_ind]
        old_list_neg.remove(random_negative_image_index)
        best_matches_neg_dict[base_img_ind] = old_list_neg

        pair_images.append([image, random_negative_image])
        pair_labels.append([0])

        base_img_ind += 1

    with open(f'{path}/saved_best_matches.pkl', 'wb') as f:
        pickle.dump(best_matches_dict, f)

    with open(f'{path}/saved_best_matches_neg.pkl', 'wb') as f:
        pickle.dump(best_matches_neg_dict, f)

    return np.asarray(pair_images), np.asarray(pair_labels)


def euclidean_distance(vectors):
    from keras import backend as K
    (featA, featB) = vectors
    sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    # Blad?
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def plot_learning_history(history, model_id, hold):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'result{hold}/model{model_id}/plots/accuracy_plot.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'result{hold}/model{model_id}/plots/loss_plot.png')


def plot_learning_history_2(history, path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{path}/accuracy_plot.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{path}/loss_plot.png')