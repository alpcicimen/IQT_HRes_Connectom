import tensorflow as tf

from tensorflow.keras.layers import *


def simple_generator(input_ch, output_ch):
    model = tf.keras.Sequential()

    # model.add(layers.Dense(36, use_bias=False, input_shape=(5,5,5,36)))
    # # # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # # model.add(layers.Reshape((5, 5, 5, 36)))

    # model.add(layers.Conv3D(kernel_size=(3,3,3), filters=30, padding='same'))
    # # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # print(model.output_shape)
    
    # model.add(layers.Conv3D(kernel_size=(3,3,3), filters=28, padding='same'))
    # # model.add(layers.BatchNormalization())\n",
    # model.add(layers.LeakyReLU())

    # print(model.output_shape)
    
    # model.add(layers.Conv3D(kernel_size=(3,3,3), filters=26, padding='same'))
    # # model.add(layers.BatchNormalization())\n",
    # model.add(layers.LeakyReLU())

    # print(model.output_shape)

    # model.add(layers.Conv3D(kernel_size=(2,2,2), filters=22, padding='same'))
    # # model.add(layers.BatchNormalization())\n",
    # model.add(layers.LeakyReLU())

    # print(model.output_shape)

    # model.add(layers.Dense(36, use_bias=False, input_shape=(11,11,11,36)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Reshape((5, 5, 5, 36)))

    model.add(Conv3D(kernel_size=(3,3,3), filters=50, padding='valid', input_shape=[11,11,11,input_ch]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv3D(kernel_size=(1,1,1), filters=100, padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # model.add(Conv3D(kernel_size=(3,3,3), filters=50, padding='valid'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())

    model.add(Conv3D(kernel_size=(3,3,3), filters=output_ch, padding='valid'))
    # model.add(layers.BatchNormalization())
    # model.add(LeakyReLU())

    print(model.output_shape)

    return model

def simple_discriminator(output_ch):
    model = tf.keras.Sequential()
    model.add(Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding='same',
                                     input_shape=[5, 5, 5, output_ch]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # model.add(layers.Dropout(0.3))

    # print(model.input_shape, model.output_shape)\n",

    model.add(Conv3D(128, (5, 5, 5), strides=(2, 2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # model.add(layers.Dropout(0.3))

    # print(model.output_shape)\n",

    model.add(Flatten())
    model.add(Dense(1))

    print(model.output_shape)

    return model

def unet3d(input_layer, start_neurons):

    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling3D((2, 2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(pool4)
    convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv3DTranspose(start_neurons * 8, (3, 3, 3), strides=(2, 2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv3DTranspose(start_neurons * 4, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv3DTranspose(start_neurons * 2, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv3DTranspose(start_neurons * 1, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv3D(22, (1, 1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer




# input_layer = Input((img_size_target, img_size_target, 1))
# output_layer = build_model(input_layer, 16)