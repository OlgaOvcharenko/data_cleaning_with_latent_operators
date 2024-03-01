import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, BatchNormalization, LeakyReLU, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras import Input, Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
#import tensorflow_probability as tfp
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x
from latent_operator_train_loop import LatentTrainLoop#, UnshifterLatentTrainLoop

def create_and_train_LOP(train_dataset, val_dataset, x_dim, z_dim, K, K2, T, epochs= 10, lr = 0.001, model_name="test"):

    base_path = f'./MODELS/{model_name}/{T}_{x_dim}_{z_dim}_{K}_epochs_{epochs}/'

    input_tuple = Input(shape=(x_dim,))


    latent_vectors = []
    
    for i in range(x_dim):
        z = Dense(z_dim, use_bias=False, activation="linear")(input_tuple)
        #latent_vectors.append(BatchNormalization()(z))
        
        #z = Dense(64, activation="relu")(input_tuple)
        #z = Dense(32, activation="relu")(z)
        #z = Dense(z_dim, activation="linear")(z)

        
        latent_vectors.append(z)
        

    encoder = Model(inputs = input_tuple,
                    outputs = latent_vectors)#[btz, btz_2])

    column_operators = []

    for i in range(x_dim):
        #d_ = Dense(32, activation = "relu")(latent_vectors[i])

        #d_bt = latent_vectors[i]
        #d_bt = Dense(12, activation = "relu")(latent_vectors[i])
        #d_bt = Dense(512, activation = "relu")(latent_vectors[i])
        d_bt = Dense(128, activation = "relu")(latent_vectors[i])
        
        #d_bt = BatchNormalization()(d_)
        #d_bt = Dropout(0.3)(d_bt)
        #d_bt = Dense(256, activation = "relu")(d_bt)
        #d_bt = BatchNormalization()(d_)
        #d_bt = Dropout(0.3)(d_bt)
        #d_bt = Dense(128, activation = "relu")(d_bt)
        #d_bt = BatchNormalization()(d_)
        #d_bt = d_
        #d_bt = Dropout(0.3)(d_bt)


        
        
        d_out = Dense(1)(d_bt) #1 column


       # d_ = Dense(64, activation = "relu")(latent_vectors[i])
       # d_bt = BatchNormalization()(d_)
       # d_2 = Dense(64, activation = "relu")(d_bt)
       # d_bt2 = BatchNormalization()(d_2)
       # d_3 = Dense(64, activation = "relu")(d_bt2)
       # d_bt3 = BatchNormalization()(d_3)
       # d_out = Dense(1)(d_bt3) #1 column
  

        column_operators.append(d_out)

    decoder = Model(inputs = latent_vectors,
                    outputs = column_operators, name="my_latops")
    
    concatenation  =  Concatenate()(column_operators)
    
    #subdivide the vertical and horizontal dimension into buckets 
    interval_size = 1 # int(z_dim / K1)

    
    if not os.path.isdir(base_path):

        #Learning rate decay===============================================
        #decay_steps = 1000
        #cosi_lr = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps)
        #optimizer = Adam(learning_rate=cosi_lr, epsilon=1e-06)
        optimizer = Adam(learning_rate=lr, epsilon=1e-06)
        #==================================================================

        train_acc_metric = tf.keras.metrics.MeanSquaredError()
        val_acc_metric = tf.keras.metrics.MeanSquaredError()

        autoencoder = Model(inputs = input_tuple, outputs = [concatenation, latent_vectors])

        
        #MANUAL TRAIN LOOP========================================================
        #print("True latent size: ", interval_size)
        LOP = LatentOperator(K, x_dim, K2, interval_size)
        
        L = LatentTrainLoop(autoencoder, encoder, decoder, LOP, epochs, optimizer, train_acc_metric, val_acc_metric, T)
        L.train_loop(train_dataset, val_dataset)

        
        os.makedirs(base_path)
        encoder.save_weights(f'{base_path}encoder.ckpt')
        decoder.save_weights(f'{base_path}decoder.ckpt')


        print(encoder.summary())
        
        return encoder, decoder, LOP, float(L.train_acc), float(L.val_acc)

        
    else:
        encoder.load_weights(f'{base_path}encoder.ckpt')
        decoder.load_weights(f'{base_path}decoder.ckpt')
        LOP = LatentOperator(K, x_dim, K2, interval_size)


        return encoder, decoder, LOP, 0.0, 0.0


def create_and_train_classifier(train_dataset, val_dataset, inp_dim, z_dim, T, n_epochs= 100, lr = 0.001, model_name="test"):
    base_path = f'./MODELS/{model_name}/Classifier_{T}_{z_dim}_{n_epochs}/'
    nnreg_model = Sequential()
    
    #nnreg_model.add(Dense(z_dim, input_dim = inp_dim))
    #nnreg_model.add(Dense(64, input_dim = inp_dim))   #better to be independent of the latent dimensionality 
    nnreg_model.add(Dense(128, kernel_initializer='normal', activation='relu', input_dim = inp_dim))
    nnreg_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    nnreg_model.add(Dense(1))

    nnreg_model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    if not os.path.isdir(base_path):
        nnreg_model.fit(train_dataset, epochs = n_epochs, verbose = 0)
        nnreg_model.save_weights(f'{base_path}classifier.ckpt')
    else:
        nnreg_model.load_weights(f'{base_path}classifier.ckpt')

    return nnreg_model
