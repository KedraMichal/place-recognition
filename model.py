from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten


def build_siamese_model():
    inputs = Input((64, 64, 3))
    x = Conv2D(filters=64, kernel_size=(2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(128)(pooledOutput)
    model = Model(inputs, outputs)

    return model


def build_siamese_model_2():
    inputs = Input((64, 64, 3))
    x = Conv2D(filters=64,kernel_size=(2,2),padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=64, kernel_size=(2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=128, kernel_size=(2,2),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=128, kernel_size=(2,2),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=256, kernel_size=(2,2),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = Dropout(0.2)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(256)(pooledOutput)
    model = Model(inputs, outputs)

    return model


def build_siamese_model_3():
    inputs = Input((64, 64, 3))
    x = Conv2D(filters=64,kernel_size=(3, 3),padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(filters=256, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(256)(pooledOutput)
    model = Model(inputs, outputs)

    return model


