import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.activations import * 
from tensorflow.keras.initializers import Constant
from tensorflow.nn import space_to_depth, depth_to_space

import keras.backend as K

class VariationalDropout(Layer):
    def __init__(self, rate=0.6, **kwargs):
        super(VariationalDropout, self).__init__(**kwargs)
        self.rate = rate
        self.log_alpha = self.add_weight(name='log_alpha',
                                 shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
    
    def call(self, inputs, training=None):
        if training:

            alpha = K.exp(self.log_alpha)
            
            random_tensor = K.random_uniform(shape=K.shape(inputs))
            dropout_prob = alpha / (alpha + 1.)
            keep_prob = 1. - dropout_prob
            
            random_tensor += keep_prob
            binary_tensor = tf.floor(random_tensor)
            
            output = inputs * binary_tensor
            output /= keep_prob
            
            return output

            # sigma = ( 1 - self.rate ) / self.rate

            # outputs = inputs * (1. + sigma * tf.random.normal(tf.shape(inputs)))
            
            # return outputs

        return inputs

class DepthToSpaceLayer(Layer):

  def __init__(self, upsampling_rate=2):
      super().__init__()
      self.upsampling_rate=upsampling_rate

  def call(self, inputs):

    batch_size, dim_i, dim_j, dim_k, c = K.int_shape(inputs)

    assert (c % (self.upsampling_rate**3) == 0) and (c > 0) # Number must be exactly divisible by 8

    if batch_size is None:
        batch_size = -1
    
    dim_i_r = dim_i * self.upsampling_rate
    dim_j_r = dim_j * self.upsampling_rate
    dim_k_r = dim_k * self.upsampling_rate
      
    oc = c // (self.upsampling_rate**3)

    out = K.reshape(inputs, (batch_size, dim_i, dim_j, dim_k, self.upsampling_rate, self.upsampling_rate, self.upsampling_rate, oc))
    out = K.permute_dimensions(out, (0, 1, 4, 2, 5, 3, 6, 7))
    out = K.reshape(out, (batch_size, dim_i_r, dim_j_r, dim_k_r, oc))
    return out

class SpaceToDepthLayer(Layer):

  def __init__(self, upsampling_rate=2):
      super().__init__()
      self.upsampling_rate=upsampling_rate

  def call(self, inputs):

    batch_size, dim_i, dim_j, dim_k, c = K.int_shape(inputs)

    assert (dim_i % (self.upsampling_rate) == 0) # Number must be exactly divisible by 8
    assert (dim_j % (self.upsampling_rate) == 0) # Number must be exactly divisible by 8
    assert (dim_k % (self.upsampling_rate) == 0) # Number must be exactly divisible by 8
      
    if batch_size is None:
        batch_size = -1
    
    dim_i_r = dim_i // self.upsampling_rate
    dim_j_r = dim_j // self.upsampling_rate
    dim_k_r = dim_k // self.upsampling_rate
      
    oc = c * (self.upsampling_rate**3)

    out = K.reshape(inputs, (batch_size, dim_i_r, self.upsampling_rate,
                                         dim_j_r, self.upsampling_rate, 
                                         dim_k_r, self.upsampling_rate, c))
    out = K.permute_dimensions(out, (0, 1, 3, 5, 2, 4, 6, 7))
    out = K.reshape(out, (batch_size, dim_i_r, dim_j_r, dim_k_r, oc))
    return out

def crop_and_or_concat_basic(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)

    # offsets for the top left corner of the crop
    offsets = [0,
               (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2,
               (x1_shape[3] - x2_shape[3]) // 2,
               0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)

    y = tf.concat([x1_crop, x2], 4)

    return y

def simple_generator(input_ch, output_ch, ipatch_size=11, f_num=50, layer_num = 1, ds=2):

    input_layer = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input')
    input_layer_t1 = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, 64], name='input_t1')

    initializer = tf.random_normal_initializer(0., 0.02)
    
    model = tf.keras.Sequential()

    model.add(Conv3D(kernel_size=(3,3,3), filters=f_num, padding='valid'))
    model.add(ReLU())

    for n in range(layer_num):

        # fn = f_num

        if n == 0:
            k = 1
        else:
            # fn = 2*f_num
            k = 3
    
        model.add(Conv3D(kernel_size=(k,k,k), filters=2*f_num, padding='valid'))
        model.add(ReLU())

    model.add(Conv3D(kernel_size=(3,3,3), filters=ds**3*output_ch, padding='valid'))
    
    model.add(DepthToSpaceLayer(ds))

    return tf.keras.Model([input_layer, input_layer_t1], model(input_layer))

def simple_generator_mmodal(input_ch, output_ch, ipatch_size=11, f_num=50, layer_num = 1, ds=2):

    input_layer = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input')
    input_layer_t1 = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, 64], name='input_t1')

    initializer = tf.random_normal_initializer(0., 0.02)
    
    model_dt = tf.keras.Sequential()
    model_t1 = tf.keras.Sequential()

    model_dt.add(Conv3D(kernel_size=(3,3,3), filters=f_num, padding='valid'))
    model_dt.add(ReLU())

    model_t1.add(Conv3D(kernel_size=(3,3,3), filters=2*f_num, padding='valid'))
    model_t1.add(ReLU())

    for n in range(layer_num):

        # fn = f_num

        if n == 0:
            k = 1
        else:
            # fn = 2*f_num
            k = 3
    
        model_dt.add(Conv3D(kernel_size=(k,k,k), filters=2*f_num, padding='valid'))
        model_dt.add(ReLU())

        model_t1.add(Conv3D(kernel_size=(k,k,k), filters=2*f_num, padding='valid'))
        model_t1.add(ReLU())

    concat_layer = concatenate([model_dt(input_layer), model_t1(input_layer_t1)])
    
    output_layer = Conv3D(kernel_size=(3,3,3), filters=ds**3*output_ch, padding='valid')(concat_layer)
    
    output_layer = DepthToSpaceLayer(ds)(output_layer)

    return tf.keras.Model([input_layer, input_layer_t1], output_layer)

def simple_generator_mmodal_2(input_ch, output_ch, ipatch_size=11, f_num=100, layer_num = 1, ds=2):

    input_layer = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input')
    input_layer_t1 = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, 64], name='input_t1')

    initializer = tf.random_normal_initializer(0., 0.02)
    
    model = tf.keras.Sequential()

    model.add(Conv3D(kernel_size=(3,3,3), filters=f_num, padding='valid'))
    model.add(ReLU())

    for n in range(layer_num):

        # fn = f_num

        if n == 0:
            k = 1
        else:
            # fn = 2*f_num
            k = 3
    
        model.add(Conv3D(kernel_size=(k,k,k), filters=2*f_num, padding='valid'))
        model.add(ReLU())

    
    model.add(Conv3D(kernel_size=(3,3,3), filters=ds**3*output_ch, padding='valid'))
    
    model.add(DepthToSpaceLayer(ds))

    return tf.keras.Model([input_layer, input_layer_t1], model(concatenate([input_layer, input_layer_t1])))

# def simple_generator(input_ch, output_ch, ipatch_size=11, f_num=50, ds=2):
#     initializer = tf.random_normal_initializer(0., 0.02)
    
#     input_layer = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input')

#     convm = Conv3D(f_num, (3, 3, 3), kernel_initializer=initializer, padding="valid")(input_layer)
    
#     convm = ReLU()(convm)
#     convm = Conv3D(f_num*2, (1, 1, 1), kernel_initializer=initializer, padding="valid")(convm)
#     convm = crop_and_or_concat_basic(input_layer, convm)
#     convm = VariationalDropout()(convm)

#     output_layer = Conv3D(output_ch*ds**3, (3, 3, 3), padding="valid")(convm)

#     # output_layer = DepthToSpaceLayer(ds)(output_layer)

#     return tf.keras.Model(input_layer, output_layer)

def simple_generator_map(input_ch = 22, output_ch = 22*8, ipatch_size=11, f_num=256):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    input_layer_dt = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input_dt')

    conv_dt = Conv3D(f_num, (5, 5, 5), kernel_initializer=initializer, padding="valid")(input_layer_dt)
    conv_dt = ReLU()(conv_dt)

    conv_dt = Conv3D(f_num, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)
    
    conv_dt = Conv3D(f_num//2, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)

    conv_dt = Conv3D(f_num//2, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)

    conv_dt = Conv3D(f_num//4, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)

    output_layer = Conv3D(output_ch*8, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)

    output_layer = DepthToSpaceLayer(2)(output_layer)

    return tf.keras.Model(input_layer_dt, output_layer)

def simple_multimodal_generator_map(input_ch, output_ch = 22, ipatch_size=21, f_num=256):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    input_layer_dt = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input_dt')
    input_layer_t1 = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, 64], name='input_t1')

    conv_dt = Conv3D(f_num, (5, 5, 5), kernel_initializer=initializer, padding="valid")(input_layer_dt)
    conv_dt = ReLU()(conv_dt)

    conv_dt = Conv3D(f_num, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)
    
    conv_dt = Conv3D(f_num//2, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)

    conv_dt = Conv3D(f_num//2, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)

    conv_dt = Conv3D(f_num//4, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = ReLU()(conv_dt)

    output_layer = Conv3D(output_ch*8, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_dt)

    output_layer = DepthToSpaceLayer(2)(output_layer)

    return tf.keras.Model([input_layer_dt, input_layer_t1], output_layer)

def simple_multimodal_generator_2(input_ch, t1_ch, output_ch, ipatch_size=11, f_num=50, ds=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    input_layer_dt = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input_dt')
    input_layer_t1 = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, t1_ch], name='input_t1')

    conv_input = concatenate([input_layer_dt, input_layer_t1])

    # conv_input = input_layer_dt

    conv_dt = Conv3D(f_num*2, (3, 3, 3), kernel_initializer=initializer, padding="valid")(conv_input)
    conv_dt = VariationalDropout()(conv_dt)
    
    conv_dt = ReLU()(conv_dt)
    conv_dt = Conv3D(f_num*4, (1, 1, 1), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = crop_and_or_concat_basic(conv_input, conv_dt)
    conv_dt = VariationalDropout()(conv_dt)

    output_layer = Conv3D(output_ch*ds**3, (3, 3, 3), padding="valid")(conv_dt)

    output_layer = DepthToSpaceLayer(ds)(output_layer)

    return tf.keras.Model([input_layer_dt, input_layer_t1], output_layer)

def simple_multimodal_generator(input_ch, t1_ch, output_ch, ipatch_size=11, f_num=50, ds=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    input_layer_dt = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, input_ch], name='input_dt')
    input_layer_t1 = tf.keras.layers.Input(shape=[ipatch_size, ipatch_size, ipatch_size, t1_ch], name='input_t1')

    conv_dt = Conv3D(f_num, (3, 3, 3), kernel_initializer=initializer, padding="valid")(input_layer_dt)

    
    conv_dt = ReLU()(conv_dt)
    conv_dt = Conv3D(f_num*2, (1, 1, 1), kernel_initializer=initializer, padding="valid")(conv_dt)
    conv_dt = crop_and_or_concat_basic(input_layer_dt, conv_dt)
    conv_dt = VariationalDropout()(conv_dt)
    
    conv_t1 = Conv3D(f_num*2, (3, 3, 3), kernel_initializer=initializer, padding="valid")(input_layer_t1)

    
    conv_t1 = ReLU()(conv_t1)
    conv_t1 = Conv3D(f_num*2, (1, 1, 1), kernel_initializer=initializer, padding="valid")(conv_t1)
    conv_t1 = crop_and_or_concat_basic(input_layer_t1, conv_t1)
    conv_t1 = VariationalDropout()(conv_t1)

    convm = conv_dt
    
    convm = concatenate([convm, conv_t1])

    output_layer = Conv3D(output_ch*ds**3, (3, 3, 3), padding="valid")(convm)

    output_layer = DepthToSpaceLayer(2)(output_layer)

    return tf.keras.Model([input_layer_dt, input_layer_t1], output_layer)

def simple_discriminator(output_ch, opatch_size = 5, f_num = 50, layer_num = 4, ds=2):
    model = tf.keras.Sequential()

    initializer = tf.random_normal_initializer(0., 0.02)

    hr_input = tf.keras.layers.Input(shape=[opatch_size*ds, opatch_size*ds, opatch_size*ds, output_ch], name='input')

    lr_input = tf.keras.layers.Input(shape=[opatch_size, opatch_size, opatch_size, output_ch], name='input_lr') # Check the lr opatchxopatch range

    hr_input_shuf = SpaceToDepthLayer(ds)(hr_input)

    input = concatenate([hr_input_shuf, lr_input])

    conv = Conv3D(kernel_size=(3,3,3), filters=f_num, padding='valid', kernel_initializer=initializer, use_bias=True, bias_initializer=Constant(1e-2))(input)
    # conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    for n in range(layer_num):

        conv = Conv3D(kernel_size=(3,3,3), filters=f_num*(2**(n+1)), padding='same', kernel_initializer=initializer, use_bias=True, bias_initializer=Constant(1e-2))(conv)
        # conv = BatchNormalization()(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
    
    conv = Conv3D(kernel_size=(3,3,3), filters=1, padding='valid', kernel_initializer=initializer)(conv)

    output_layer = Flatten()(conv)
    output_layer = Dense(1)(output_layer)

    # return tf.keras.Model(input, output_layer)
    return tf.keras.Model([hr_input, lr_input], output_layer)

def downsample(filters, size, apply_batchnorm=False, repconvs=0):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  for i in range(repconvs):
      result.add(
          tf.keras.layers.Conv3D(filters, size, strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    
      # result.add(tf.keras.layers.BatchNormalization())
    
      result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=True, repconvs=0):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()

  # result.add(tf.keras.layers.Conv3DTranspose(filters, size, strides=2,
  #                                           padding='same',
  #                                           kernel_initializer=initializer,
  #                                           use_bias=False))

  result.add(UpSampling3D(size=(2,2,2)))

  result.add(tf.keras.layers.Conv3D(filters, size, strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.2))

  result.add(tf.keras.layers.ReLU())

  for i in range(repconvs):
      result.add(
          tf.keras.layers.Conv3D(filters, size, strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    
      result.add(tf.keras.layers.ReLU())
  return result

def ResBlock(inputs, num_filters, filter_size=3):

    x = Conv3D(num_filters, filter_size, padding="same", activation="relu")(inputs)
    x = Conv3D(num_filters, filter_size, padding="same")(x)
    x = Add()([inputs, x])
    
    return x

def gen_pix2pix(input_shape = [16, 16, 16, 14], filter_size=3, output_ch=22):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    down_stack = [
    downsample(100, filter_size, apply_batchnorm=False),  # (batch_size, 16, 16, 16, 64)
    downsample(200, filter_size, apply_batchnorm=False),  # (batch_size, 8, 8, 8, 128)
    downsample(400, filter_size, apply_batchnorm=False),  # (batch_size, 4, 4, 4, 256)
    downsample(400, filter_size, apply_batchnorm=False),  # (batch_size, 2, 2, 2, 256)
    ]
    
    up_stack = [
    upsample(400, filter_size),  # (batch_size, 4, 4, 4, 512)
    upsample(200, filter_size),  # (batch_size, 8, 8, 8, 256)
    upsample(100, filter_size),  # (batch_size, 16, 16, 16, 128)
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3DTranspose(output_ch, 2*filter_size,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer)  # (batch_size, 256, 256, 3) 32
        
    x = inputs
    
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    # x = Conv3D(output_ch, filter_size, padding='same', kernel_initializer=initializer)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def gen_pix2pix_mmodal(input_shape = [16, 16, 16, 6],
                       input_shape_t1 = [16, 16, 16, 8], filter_size=3, output_ch=22):
    inputs = tf.keras.layers.Input(shape=input_shape)

    inputs_t1 = tf.keras.layers.Input(shape=input_shape_t1)
    
    down_stack = [
        downsample(50, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 16, 16, 16, 64)
        downsample(200, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 8, 8, 8, 128)
        downsample(800, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 4, 4, 4, 256)
        downsample(800, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 2, 2, 2, 256)
    ]

    down_stack_t1 = [
        downsample(50, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 16, 16, 16, 64)
        downsample(200, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 8, 8, 8, 128)
        downsample(800, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 4, 4, 4, 256)
        downsample(800, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 2, 2, 2, 256)
        downsample(800, filter_size, repconvs=0, apply_batchnorm=False),  # (batch_size, 2, 2, 2, 256)
    ]
    
    up_stack = [
        upsample(800, filter_size, repconvs=0),  # (batch_size, 4, 4, 4, 512)
        upsample(200, filter_size, repconvs=0),  # (batch_size, 8, 8, 8, 256)
        upsample(50, filter_size, repconvs=0),  # (batch_size, 16, 16, 16, 128)
    ]

    upsamples = [800,200,50]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3DTranspose(output_ch, 2*filter_size,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer)  # (batch_size, 256, 256, 3) 32

    # last = DepthToSpaceLayer(2)
        
    x = inputs
    x_t1 = inputs_t1
    
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        # print(x)
        skips.append(x)

    for down in down_stack_t1:
        x_t1 = down(x_t1)
    
    skips = reversed(skips[:-1])

    x = tf.keras.layers.Add()([x, x_t1])

    # for _ in range(2):
    #     x = ResBlock(x, 800)

    # unum = 0
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)
    
    return tf.keras.Model(inputs=[inputs, inputs_t1], outputs=x)

def disc_pix2pix(input_shape = [32, 32, 32, 6], filter_size=3):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    # tar = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]//8 + 8], name='target_image')
    tar = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]//8], name='target_image')
    
    
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 32, 32, channels*2)
    
    down1 = downsample(100, filter_size, False)(x)  # (batch_size, 16, 16, 64)
    down2 = downsample(200, filter_size, False)(down1)  # (batch_size, 8, 8, 128)
    down3 = downsample(400, filter_size, False)(down2)  # (batch_size, 4, 4, 256)

    # last = tf.keras.layers.Conv3D(1, filter_size, strides=1,
    #                             kernel_initializer=initializer)(down3)  # (batch_size, 2, 2, 1)

    
    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (batch_size, 6, 6, 256)
    conv = tf.keras.layers.Conv3D(400, filter_size, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 3, 3, 512)
    
    # batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(conv)
    
    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (batch_size, 5, 5, 512)
    
    last = tf.keras.layers.Conv3D(1, filter_size, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 2, 2, 1)

    # last = Dense(1)(Flatten()(last))
    
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def downsample_channel(n, prev_layer):
    initializer = tf.random_normal_initializer(0., 0.02)

    # conv = Dropout(.4)(prev_layer)

    conv = Conv3D(n, (3, 3, 3), strides=2, padding="same")(prev_layer)
    conv = BatchNormalization(axis=-1, trainable=True)(conv)
    conv = ReLU()(conv)

    conv = Conv3D(n, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv)
    conv = BatchNormalization(axis=-1, trainable=True)(conv)
    conv = ReLU()(conv)
    conv = Conv3D(n, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv)
    conv = BatchNormalization(axis=-1, trainable=True)(conv)
    conv = ReLU()(conv)

    return conv

def upsample_channel(n, prev_layer, concat_layer, concat=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    deconv = Conv3DTranspose(n, (6, 6, 6), kernel_initializer=initializer, strides=2, padding="same")(prev_layer)
    
    if concat:
        uconv = concatenate([deconv, concat_layer], 4)
        uconv = Conv3D(n, (3, 3, 3), kernel_initializer=initializer, padding="same")(uconv)
    else:
        uconv = Conv3D(n, (3, 3, 3), kernel_initializer=initializer, padding="same")(deconv)

    uconv = ReLU()(uconv)

    return uconv

def unet3d(input_layer, start_neurons, output_neurons):
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # DownConv Layers
    
    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(input_layer)
    conv1 = BatchNormalization(axis=-1, trainable=True)(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv1)
    conv1 = BatchNormalization(axis=-1, trainable=True)(conv1)
    conv1 = ReLU()(conv1)

    conv2 = downsample_channel(start_neurons * 2, conv1)
    conv3 = downsample_channel(start_neurons * 4, conv2)
    conv4 = downsample_channel(start_neurons * 8, conv3)

    # Middle Layer
    
    convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", strides=2, padding="same")(conv4)
    convm = BatchNormalization(axis=-1, trainable=True)(convm)
    convm = ReLU()(convm)
    convm = Conv3D(start_neurons * 16, (3, 3, 3), padding="same")(convm)
    convm = BatchNormalization(axis=-1, trainable=True)(convm)
    convm = ReLU()(convm)
    
    # UpConv Layers

    uconv4 = upsample_channel(start_neurons * 8, convm, conv4)
    uconv3 = upsample_channel(start_neurons * 4, uconv4, conv3)
    uconv2 = upsample_channel(start_neurons * 2, uconv3, conv2)
    uconv1 = upsample_channel(start_neurons * 1, uconv2, conv1)

    output_layer = Conv3D(output_neurons*8, (6, 6, 6), padding="same")(uconv1)

    output_layer = DepthToSpaceLayer(2)(output_layer)

    return output_layer

def unet3d_dti(patch_size, start_neurons, output_neurons):
# def multimodalunet3d(patch_size, start_neurons, output_neurons):

    initializer = tf.random_normal_initializer(0., 0.02)
    
    input_layer_dt = tf.keras.layers.Input(shape=[patch_size, patch_size, patch_size, 6], name='input_dt')
    input_layer_t1 = tf.keras.layers.Input(shape=[2*patch_size, 2*patch_size, 2*patch_size, 8], name='input_t1')
    
    # DownConv Layers
    
    conv1_dt = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(input_layer_dt) # (32,32,32,6) -> (32,32,32,32)
    conv1_dt = ReLU()(conv1_dt)
    conv1_dt = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv1_dt)
    conv1_dt = ReLU()(conv1_dt)

    conv2_dt = downsample_channel(start_neurons * 2, conv1_dt) # (32,32,32,32) -> (16,16,16,64) 16 -> 8
    conv3_dt = downsample_channel(start_neurons * 4, conv2_dt) # (16,16,16,64) -> (8,8,8,256) 8 -> 4
    conv4_dt = downsample_channel(start_neurons * 8, conv3_dt) # (8,8,8,256) -> (4,4,4,512) 4 -> 2
    
    # conv1_t1 = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(input_layer_t1)
    # conv1_t1 = ReLU()(conv1_t1)
    # conv1_t1 = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv1_t1) 
    # conv1_t1 = ReLU()(conv1_t1)

    # conv1_t1 = downsample_channel(start_neurons * 2, conv1_t1) # Downsample one level more
    # conv2_t1 = downsample_channel(start_neurons * 4, conv1_t1) # (32,32,32,32) -> (16,16,16,64)
    # conv3_t1 = downsample_channel(start_neurons * 6, conv2_t1) # (16,16,16,64) -> (8,8,8,256)
    # conv4_t1 = downsample_channel(start_neurons * 8, conv3_t1) # (8,8,8,256) -> (4,4,4,512)

    # Middle Layer

    # convm = concatenate([conv4_dt, conv4_t1]) # Latent Space (4,4,4,1024)
    
    convm = Conv3D(start_neurons * 16, (3, 3, 3), kernel_initializer=initializer, strides=2, padding="same")(conv4_dt)
    convm = ReLU()(convm)
    convm = Conv3D(start_neurons * 16, (3, 3, 3), kernel_initializer=initializer, padding="same")(convm)
    convm = ReLU()(convm)
    
    # UpConv Layers

    uconv4 = upsample_channel(start_neurons * 8, convm, conv4_dt)
    uconv3 = upsample_channel(start_neurons * 4, uconv4, conv3_dt)
    uconv2 = upsample_channel(start_neurons * 2, uconv3, conv2_dt)
    uconv1 = upsample_channel(start_neurons * 1, uconv2, conv1_dt)

    output_layer = Conv3D(output_neurons*8, (3, 3, 3), kernel_initializer=initializer, padding="same")(uconv1)

    output_layer = DepthToSpaceLayer(2)(output_layer)

    return tf.keras.Model([input_layer_dt, input_layer_t1], output_layer)

def multimodalunet3d(patch_size, start_neurons, output_neurons):

    initializer = tf.random_normal_initializer(0., 0.02)
    
    input_layer_dt = tf.keras.layers.Input(shape=[patch_size, patch_size, patch_size, 6], name='input_dt')
    input_layer_t1 = tf.keras.layers.Input(shape=[2*patch_size, 2*patch_size, 2*patch_size, 8], name='input_t1')
    
    # DownConv Layers
    
    conv1_dt = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(input_layer_dt) # (32,32,32,6) -> (32,32,32,32)
    conv1_dt = ReLU()(conv1_dt)
    conv1_dt = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv1_dt)
    conv1_dt = ReLU()(conv1_dt)

    conv2_dt = downsample_channel(start_neurons * 2, conv1_dt) # (32,32,32,32) -> (16,16,16,64) 16 -> 8
    conv3_dt = downsample_channel(start_neurons * 4, conv2_dt) # (16,16,16,64) -> (8,8,8,256) 8 -> 4
    conv4_dt = downsample_channel(start_neurons * 8, conv3_dt) # (8,8,8,256) -> (4,4,4,512) 4 -> 2
    
    conv1_t1 = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(input_layer_t1)
    conv1_t1 = ReLU()(conv1_t1)
    conv1_t1 = Conv3D(start_neurons * 1, (3, 3, 3), kernel_initializer=initializer, padding="same")(conv1_t1) 
    conv1_t1 = ReLU()(conv1_t1)

    conv1_t1 = downsample_channel(start_neurons * 2, conv1_t1) # Downsample one level more
    conv2_t1 = downsample_channel(start_neurons * 4, conv1_t1) # (32,32,32,32) -> (16,16,16,64)
    conv3_t1 = downsample_channel(start_neurons * 6, conv2_t1) # (16,16,16,64) -> (8,8,8,256)
    conv4_t1 = downsample_channel(start_neurons * 8, conv3_t1) # (8,8,8,256) -> (4,4,4,512)

    # Middle Layer

    convm = concatenate([conv4_dt, conv4_t1]) # Latent Space (4,4,4,1024)
    
    convm = Conv3D(start_neurons * 16, (3, 3, 3), kernel_initializer=initializer, strides=2, padding="same")(convm)
    convm = ReLU()(convm)
    convm = Conv3D(start_neurons * 16, (3, 3, 3), kernel_initializer=initializer, padding="same")(convm)
    convm = ReLU()(convm)
    
    # UpConv Layers

    uconv4 = upsample_channel(start_neurons * 8, convm, conv4_dt)
    uconv3 = upsample_channel(start_neurons * 4, uconv4, conv3_dt)
    uconv2 = upsample_channel(start_neurons * 2, uconv3, conv2_dt)
    uconv1 = upsample_channel(start_neurons * 1, uconv2, conv1_dt)

    output_layer = Conv3D(output_neurons*8, (3, 3, 3), kernel_initializer=initializer, padding="same")(uconv1)

    output_layer = DepthToSpaceLayer(2)(output_layer)

    return tf.keras.Model([input_layer_dt, input_layer_t1], output_layer)

