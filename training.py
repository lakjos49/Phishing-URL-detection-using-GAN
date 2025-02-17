import random

import numpy as np
import pandas as pd
import tensorflow as tf

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Importing neural network modules
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

data=pd.read_csv("final_dataset.csv")

data=data.drop(['url'],axis=1)
data_legit=data[data['label']==0]
data_phising=data[data['label']==1]

X=data.drop("label",axis=1)
Y=data.label


#generator
input_dim = X.shape[1]
print("Input dimension for the model:", input_dim)
def build_generator(input_dim):
  model=Sequential()
  model.add(Dense(32,input_dim=input_dim,activation="relu",kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Dense(64,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dense(128,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dense(10,activation="linear"))
  model.summary()
  return model

def build_discriminator():
  model=Sequential()
  model.add(Dense(128,input_dim=X.shape[1],activation="relu",kernel_initializer='he_uniform'))
  model.add(Dense(64,activation="relu"))
  model.add(Dense(32,activation="relu"))
  model.add(Dense(32,activation="relu"))
  model.add(Dense(16,activation="relu"))
  model.add(Dense(1,activation="sigmoid"))
  opt = Adam(learning_rate=0.000017, beta_1=0.01)
  model.compile(optimizer=opt,loss='binary_crossentropy')
  model.summary()
  return model

def build_gan(generator,discriminator):
  discriminator.trainable=False
  gan_input=Input(shape=(generator.input_shape[1],))
  x=generator(gan_input)
  gan_output=discriminator(x)
  gan=Model(gan_input,gan_output)
  gan.summary()
  return gan

def generate_synthetic_data(generator,num_samples):
    noise=np.random.normal(0,1,(num_samples,generator.input_shape[1]))
    attacked_data=generator.predict(noise)
    return attacked_data


generator = build_generator(input_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
opt = Adam(learning_rate=0.001, beta_1=0.9)
gan.compile(optimizer=opt, loss='binary_crossentropy')
num_epochs = 500
batch_size = 64
half_batch = int(batch_size / 2)

for epoch in range(num_epochs):
    X_syn = generate_synthetic_data(generator, half_batch)
    y_syn = np.zeros((half_batch, 1))
    X_real = data_phising.drop("label", axis=1).sample(half_batch)
    y_real = np.ones((half_batch, 1))

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(X_real, y_real)
    d_loss_syn = discriminator.train_on_batch(X_syn, y_syn)
    noise = np.random.normal(0, 1, (batch_size, 10))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    if epoch % 10 == 0 or epoch==num_epochs-1:
        d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)
        print(f'Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')


#now we will use the train generator to create 2858 syn phish url data
synthetic_data=generate_synthetic_data(generator,5715)
# make sure synthetic_data has no of columns = no of columns of data_phising-1(label column)
# Make sure synthetic_data has the correct number of columns
if synthetic_data.shape[1] == data_phising.shape[1] - 1:
    column_names = data_phising.drop("label", axis=1).columns  # Use column names without 'label'
    df = pd.DataFrame(synthetic_data, columns=column_names)

    # Add the 'label' column
    df['label'] = "SYN_phish"

    # Create df2 to match df structure
    df2 = data_phising.drop("label", axis=1)
    df2['label'] = "Real_phish"

    # Combine the DataFrames
    combined_df = pd.concat([df, df2])
else:
    print("Error: The number of columns in synthetic_data does not match the expected number.")
merged_df=combined_df
merged_df.to_csv('merged_dataset.csv', index=False)


