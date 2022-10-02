import numpy as np
#import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import json
import pickle
import tensorflow.keras.backend as K


def get_full_index(images, labels):
    unique_labels = np.unique(labels)

    full_index_list = []
    for ind_lab in unique_labels:
        ind_list = []
        for ind, lab in enumerate(labels):
            if lab == ind_lab:
                ind_list.append(ind)
        full_index_list.append(ind_list)

    return full_index_list


def get_triplet(images, labels, full_index_list, path, save_json=True):
    try:
        with open(f'{path}/triplets.json', 'rb') as f:
            if save_json==True:
                triplets = pickle.load(f)
            else:
                triplets = []
    except:
        triplets = []

    random_ind = np.random.randint(0, len(labels)-1)
    _, rand_lab = images[random_ind], labels[random_ind]
    if rand_lab == 0:
        print(rand_lab)
    if rand_lab==1 or rand_lab==2 or rand_lab==4 or rand_lab==8:
        rand_lab = 3

    positive_images = full_index_list[rand_lab]

    while True:
        random_positive_image_indexes = np.random.choice(positive_images, 2, replace=False)
        first_elem, sec_elem = random_positive_image_indexes[0], random_positive_image_indexes[1]
        if first_elem <= sec_elem:
            tuple1 = (first_elem, sec_elem)
        else:
            tuple1 = (sec_elem, first_elem)
        random_positive_image_index_1 = random_positive_image_indexes[0]
        random_positive_image_index_2 = random_positive_image_indexes[1]
        random_positive_image_1 = images[random_positive_image_index_1]
        random_positive_image_2 = images[random_positive_image_index_2]

        negative_images = full_index_list[:rand_lab] + full_index_list[rand_lab + 1:]
        negative_images_flatten = [item for sublist in negative_images for item in sublist]

        random_negative_image_index = np.random.choice(negative_images_flatten)
        random_negative_image = images[random_negative_image_index]

        if (tuple1[0], tuple1[1], random_negative_image) not in triplets:
            break

    if save_json == True:
        with open(f'{path}/triplets.json', 'wb') as f:
                pickle.dump(triplets, f)

    return random_positive_image_1, random_positive_image_2, random_negative_image


def generate_triplets(images, labels, full_index_list, path, save_json=False):
    batch_size=64
    while True:
        list_a = []
        list_p = []
        list_n = []

        for i in range(batch_size):
            a, p, n = get_triplet(images, labels, full_index_list, path, save_json=save_json)
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)

        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')
        label = np.ones(batch_size)
        yield [A, P, N], label


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss


