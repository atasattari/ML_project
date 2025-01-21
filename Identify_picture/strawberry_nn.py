import numpy as np
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



def get_model(numfm, numnodes, input_shape = (181, 131, 3),
              output_size = 3):

    # Initialize the model.
    model = km.Sequential()

    # Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size = (2,2),input_shape = input_shape,activation = 'relu'))
    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size = (21, 11),strides = (10, 10)))
    
    model.add(kl.Dropout(0.5, name = 'dropout1'))
    model.add(kl.Conv2D(numfm * 2, kernel_size = (5, 1),
                        activation = 'relu'))

    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size = (2 ,2 ),strides = (2, 2)))
    
    # Convert the network from 2D to 1D.
    
    model.add(kl.Flatten())
    model.add(kl.Dropout(0.5, 
                         name = 'dropout2'))
    # Add a fully-connected layer.
    model.add(kl.Dense(numnodes,
                       activation = 'tanh'))

    # Add the output layer.
    model.add(kl.Dense(output_size, activation = 'softmax'))

    # Return the model.
    return model


if __name__ == '__main__':
    # Read the data
    print('Reading data... \n')
    data = np.load('strawberries.npz')
    x = data['x']
    y = data['y']

    # The training data goes from 0-4, but we want it to be 0-3.
    y = y-1

    # Show some sample data.
    n_pictures = 3
    fig, ax = plt.subplots(n_pictures,3,figsize=(10,4*n_pictures))
    for i in range(3):
        for j in range(n_pictures):
            ax[j,i].imshow(x[y==i][j])
            if j==0:
                ax[j,i].set_title(f'Class {i}')
    fig.tight_layout()
    plt.show()
        
    # Split the data into training and testing sets.
    y = ku.to_categorical(y)
    x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


    # Fake picture constructor.
    datagen = ImageDataGenerator(
        # Subtract the mean -> Learn the deviations...
        featurewise_center=True,
        # Normalize by the standard deviation.
        # Similar scale -> Faster convergence.
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        # scale colors to [0,1]
        rescale=1./255,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2)

    # To find the mean and std as requested in the configuration.
    print('Normalizing data... \n')
    datagen.fit(x_train)


    batch_size = 70
    # Fake picture generator.
    train_generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True)
    
    model = get_model(6,50)
    model.summary()

    print('Building model... \n')
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    fit = model.fit(train_generator, epochs = 100, batch_size = 20, verbose = 2)
    score = model.evaluate(x_test, y_test)
    print(f'Score output: {score}')

    print("Options below were effective. The network doesn't use that many pictures")
    print('featurewise_center=True') # Subtract the mean -> Learn the deviations...
    print('featurewise_std_normalization=True')
    print('rescale=1./255')

    