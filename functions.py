import glob
import cv2
import numpy as np
import tensorflow.keras.backend as b
import matplotlib.pyplot as plt


def read_data(path, queries=False):
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
        img = cv2.resize(img, (96, 72))
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
        if i % 2 == 0:
            pair_images.append([image, random_positive_image])
            pair_labels.append([1])

        if i % 2 != 0:
            negative_images = full_index_list[:label] + full_index_list[label + 1:]
            negative_images_flatten = [item for sublist in negative_images for item in sublist]
            random_negative_image_index = np.random.choice(negative_images_flatten)
            random_negative_image = images[random_negative_image_index]

            pair_images.append([image, random_negative_image])
            pair_labels.append([0])
        i = i+1

    return np.asarray(pair_images), np.asarray(pair_labels)


def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = b.sum(b.square(featA - featB), axis=1, keepdims=True)

    return b.sqrt(b.maximum(sum_squared, b.epsilon()))


def plot_learning_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('result/plot_60/accuracy_plot_model_30e3.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('result/plot_60/loss_plot_model_30e3.png')