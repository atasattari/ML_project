# This builds and trains a variational autoencoder, and
# applies it to MNIST data.
#

import tensorflow as tf

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import mnist

import os



####################################################################


# The size of the latent space.
latent_dim = 100

# Get the data.
(x_train, _), (x_test, _) = mnist.load_data()

# Reshape and scale.
x_train = x_train.astype('float32').reshape(60000, 28, 28, 1) / 255.
x_test = x_test.astype('float32').reshape(10000, 28, 28, 1) / 255.


####################################################################


def sampling(args):
    
    """This function reparameterizes the random sampling which is needed
    to feed the decoding network.

    """

    # Take apart the input arguments.
    z_mean, z_log_var = args

    # Get the dimensions of the problem.
    batch_size = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    # By default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape = (batch_size, dim))

    # Return the reparameterized result.
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


####################################################################

# The Encoder

# The image inputted into the encoder.
input_image = kl.Input(shape = (28, 28, 1))

# Add a 2D convolutional layer, with 16 feature maps.
# input size = 28 x 28 x 1
# output size = 28 x 28 x 16
x = kl.Conv2D(16, kernel_size = (3, 3),
              activation = 'relu',
              padding = 'same')(input_image)

# Add a max pooling layer.
# input size 28 x 28 x 16
# output size 14 x 14 x 16
x = kl.MaxPooling2D(pool_size = (2, 2),
                    strides = (2, 2))(x)

# Add a 2D convolutional layer, with 32 feature maps.
# input size = 14 x 14 x 16
# output size = 14 x 14 x 32
x = kl.Conv2D(32, kernel_size = (3, 3),
              activation = 'relu',
              padding = 'same')(x)

# Add a max pooling layer.
# input size 14 x 14 x 32
# output size 7 x 7 x 32
x = kl.MaxPooling2D(pool_size = (2, 2),
                    strides = (2, 2))(x)

# Flatten the output so that it can be fed into the two output layers.
x = kl.Flatten()(x)

# The two output layers for the encoder.
z_mean = kl.Dense(latent_dim, activation = 'linear')(x)
z_log_var = kl.Dense(latent_dim, activation = 'linear')(x)


# Create a layer which applies the sampling function to the previous
# to inputs.
z = kl.Lambda(sampling,
              output_shape = (latent_dim,))([z_mean, z_log_var])


####################################################################

# The Decoder

# The decoder input.
decoder_input = kl.Input(shape = (latent_dim,))

# A fully-connected layer, to bulk things up to start.
x = kl.Dense(7 * 7 * 32, activation = 'relu')(decoder_input)

# Reshape to the correct starting shape.
x = kl.Reshape((7, 7, 32))(x)

# Add a 2D transpose convolutional layer, with 32 feature maps.
# input size = 7 x 7 x 32
# output size = 7 x 7 x 32
x = kl.Conv2DTranspose(32, kernel_size = (3, 3),
                       activation = 'relu',
                       padding = 'same')(x)

# Add upsampling
# input size = 7 x 7 x 32
# output size = 14 x 14 x 32
x = kl.UpSampling2D(size = (2, 2))(x)

# Add a 2D transpose convolutional layer, with 16 feature maps.
# input size = 14 x 14 x 32
# output size = 14 x 14 x 16
x = kl.Conv2DTranspose(16, kernel_size = (3, 3),
                       activation = 'relu',
                       padding = 'same')(x)

# Add upsampling
# input size = 14 x 14 x 16
# output size = 28 x 28 x 16
x = kl.UpSampling2D(size = (2, 2))(x)


# Add a 2D transpose convolutional layer, with 1 feature map.  This is
# the decoder output.
# input size = 28 x 28 x 16
# output size = 28 x 28 x 1
decoded = kl.Conv2DTranspose(1, kernel_size = (3, 3),
                             activation = 'sigmoid',
                             padding = 'same')(x)


####################################################################

# Build the models
encoder = km.Model(inputs = input_image,
                   outputs = [z_mean, z_log_var, z])
decoder = km.Model(inputs = decoder_input,
                   outputs = decoded)


output_image = decoder(encoder(input_image)[2])

vae = km.Model(inputs = input_image,
               outputs = output_image)


####################################################################


def vae_loss(input_image, output_image):
    """The variational autoencoder loss function.  This function will
    calculate the mean squared error for the resconstruction loss, and
    the KL divergence of Q(z|X).

    """

    # Calculate the KL divergence loss.
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis = -1)

    # Calculate the mean squared error, and use it for the
    # reconstruction loss.
    reconstruction_loss = 784 * K.mean(K.square(input_image - output_image))
    
    # Return the sum of the two loss functions.
    return reconstruction_loss + kl_loss


####################################################################
#import keras.optimizers as ko
#opt = ko.SGD(learning_rate = 0.00001)
# Compile the model.
vae.compile(loss = vae_loss,
            optimizer = 'adam',
            metrics = ['mean_squared_error'])

# And fit.
fit = vae.fit(x_train, x_train, epochs = 50,
              batch_size = 64,
              verbose = 2)

# Check the test data.
score = vae.evaluate(x_test, x_test)
print('score is', score)

# Save the models.
vae.save('data/mnist_vae.h5')
encoder.save('data/mnist_encoder.h5')
decoder.save('data/mnist_decoder.h5')

####################################################################
