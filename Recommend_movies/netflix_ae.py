import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

print("Reading netflix input file.")


df = pd.read_csv("ratings.sparse.small.csv")

train, test = train_test_split(df, test_size=0.2)

train = train.to_numpy(dtype=float)
test = test.to_numpy(dtype=float)


print("Building network.")
latent_dim = 5
input_data = kl.Input(shape = (4499,), name = 'Input')
x = kl.Dense(50, activation = 'relu')(input_data)
x = kl.Dropout(0.5)(x)
x = kl.Dense(40, activation = 'relu')(x)
x = kl.Dropout(0.5)(x)
x = kl.Dense(30, activation = 'relu')(x)
x = kl.Dropout(0.5)(x)
x = kl.Dense(20, activation = 'relu')(x)
x = kl.Dropout(0.5)(x)
x = kl.Dense(10, activation = 'relu')(x)
z_mean = kl.Dense(latent_dim, activation = 'linear')(x)
# encoder = km.Model(inputs = input_data, outputs = z_mean)
# encoder.summary()

decoder_input = kl.Input(shape=(latent_dim,), name='LatentInput')
x = kl.Dense(10, activation='relu')(decoder_input)
x = kl.Dropout(0.5)(x)
x = kl.Dense(20, activation = 'relu')(x)
x = kl.Dropout(0.5)(x)
x = kl.Dense(30, activation = 'relu')(x)
x = kl.Dropout(0.5)(x)
x = kl.Dense(40, activation = 'relu')(x)
x = kl.Dropout(0.5)(x)
x = kl.Dense(50, activation = 'relu')(x)

# x = kl.Dense(30, activation = 'relu')(x)
decoded = kl.Dense(4499, activation='linear')(x)

# decoder = km.Model(inputs = decoder_input, outputs = decoded)
# decoder.summary()

def loss(input_data, output_data):
    mask = tf.cast(tf.not_equal(input_data, 0), dtype=tf.float32)
    reconstruction_loss = K.sum(K.square((input_data- output_data)*mask))/K.sum(mask)
    return reconstruction_loss


# Build the models
encoder = km.Model(inputs = input_data,
                   outputs = z_mean)
decoder = km.Model(inputs = decoder_input,
                   outputs = decoded)

output_data = decoder(encoder(input_data))
model = km.Model(inputs = input_data,
               outputs = output_data)

model.compile(loss = loss, optimizer = 'adam',metrics = loss)

BUFFER_SIZE = 500
BATCH_SIZE = 100
train_dataset = tf.data.Dataset.from_tensor_slices(train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


print("Training network.")
def fake_train(current_number,train_dataset, fake_train_count = 5):
    possbile_batch = np.arange(len(train_dataset))
    mask = possbile_batch!=current_number
    random_batch_choices = possbile_batch[mask]
    random_batch_number = np.random.choice(random_batch_choices,fake_train_count)
    
    for count, batch in enumerate(train_dataset):
        if count in random_batch_number:
            f_x = model.predict(batch,verbose= 0)
            model.train_on_batch(f_x,f_x)

for epoch in range(100):
    for count, batch in enumerate(train_dataset):
        lose = model.train_on_batch(batch,batch)
        if epoch>20:
        	fake_train(count,train_dataset,5)


score = model.evaluate(train, train, verbose=0)
print('Training score:', score)

score = model.evaluate(test, test, verbose=0)
print('Test score:', score)
