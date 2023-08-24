# def build_model(input_layer, start_neurons):
#
#
#
#     conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(input_layer)
#     conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(conv1)
#     pool1 = MaxPooling3D((2, 2, 2))(conv1)
#     pool1 = Dropout(0.25)(pool1)
#
#     conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(pool1)
#     conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(conv2)
#     pool2 = MaxPooling3D((2, 2, 2))(conv2)
#     pool2 = Dropout(0.5)(pool2)
#
#     conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(pool2)
#     conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(conv3)
#     pool3 = MaxPooling3D((2, 2, 2))(conv3)
#     pool3 = Dropout(0.5)(pool3)
#
#     conv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(pool3)
#     conv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(conv4)
#     pool4 = MaxPooling3D((2, 2, 2))(conv4)
#     pool4 = Dropout(0.5)(pool4)
#
#     # Middle
#     convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(pool4)
#     convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(convm)
#
#     deconv4 = Conv3DTranspose(start_neurons * 8, (3, 3, 3), strides=(2, 2, 2), padding="same")(convm)
#     uconv4 = concatenate([deconv4, conv4])
#     uconv4 = Dropout(0.5)(uconv4)
#     uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(uconv4)
#     # uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(deconv4)
#     uconv4 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(uconv4)
#
#     deconv3 = Conv3DTranspose(start_neurons * 4, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv4)
#     uconv3 = concatenate([deconv3, conv3])
#     uconv3 = Dropout(0.5)(uconv3)
#     uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(uconv3)
#     # uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(deconv3)
#     uconv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(uconv3)
#
#     deconv2 = Conv3DTranspose(start_neurons * 2, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv3)
#     uconv2 = concatenate([deconv2, conv2])
#     uconv2 = Dropout(0.5)(uconv2)
#     uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(uconv2)
#     # uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(deconv2)
#     uconv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(uconv2)
#
#     deconv1 = Conv3DTranspose(start_neurons * 1, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv2)
#     uconv1 = concatenate([deconv1, conv1])
#     uconv1 = Dropout(0.5)(uconv1)
#     uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)
#     # uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(deconv1)
#     uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)
#
#     print(uconv1.shape)
#
#     output_layer = Conv3D(22, (1, 1, 1), padding="same", activation="sigmoid")(uconv1)
#
#     return output_layer
#
# input_l = Input((32, 32, 32, 36))
# output_layer = build_model(input_l, 22)

# %%%%%%%%%%%%%%%%%%%%%%%%%

# model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(5,5,5,36,)))
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
#
# model.add(layers.Reshape((7, 7, 256)))
# assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
#
# model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
# assert model.output_shape == (None, 7, 7, 128)
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
#
# model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
# assert model.output_shape == (None, 14, 14, 64)
# model.add(layers.BatchNormalization())
# model.add(layers.LeakyReLU())
#
# model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
# assert model.output_shape == (None, 2, 2, 2, 22)


