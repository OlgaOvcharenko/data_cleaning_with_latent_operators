import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, BatchNormalization, LeakyReLU, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras import Input, Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
#import tensorflow_probability as tfp
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x
from latent_operator_train_loop import LatentTrainLoop
from copy import deepcopy
from sklearn.metrics import mean_squared_error, accuracy_score

def predict_on_enhanced(x, LOP, encoder, decoder, _translate_1, GT_K = None):

    K = LOP.n_rotations
    Zs = encoder(x)

    # #PRINT-------------------------------------------------------
    # aa = []    
    # for i in range(x.shape[1]):
    #     a = Zs[i][-1]
    #     aa.append(tf.expand_dims(LOP.translate_operator(a, shift=1), axis = 0))

    # b = []    
    # for i in range(x.shape[1]):
    #     a = Zs[i][-1]
    #     b.append(tf.expand_dims(LOP.translate_operator(a, shift=2), axis = 0))

    # c = []    
    # for i in range(x.shape[1]):
    #     a = Zs[i][-1]
    #     c.append(tf.expand_dims(LOP.translate_operator(a, shift=3), axis = 0))

    # d = []    
    # for i in range(x.shape[1]):
    #     a = Zs[i][-1]
    #     d.append(tf.expand_dims(LOP.translate_operator(a, shift=4), axis = 0))

    # e = []    
    # for i in range(x.shape[1]):
    #     a = Zs[i][-1]
    #     e.append(tf.expand_dims(LOP.translate_operator(a, shift=5), axis = 0))

    # f = []
    # for i in range(x.shape[1]):
    #     a = Zs[i][-1]
    #     f.append(tf.expand_dims(LOP.translate_operator(a, shift=6), axis = 0))
    # g = []
    # for i in range(x.shape[1]):
    #    a = Zs[i][-1]
    #    g.append(tf.expand_dims(LOP.translate_operator(a, shift=6), axis = 0))





    # col_to_print = -1
        
    # print(x[-1],
    #       "\n",
    #       #Zs[col_to_print][-1],
    #       #"\n",
    #       decoder(tf.unstack(Zs, axis = 0))[col_to_print][-1],
    #       "\n",
    #       #aa[col_to_print],
    #      # "\n",
    #       decoder(aa)[col_to_print][0],
    #       "\n",
    #       #b[col_to_print],
    #      # "\n",
    #       decoder(b)[col_to_print][0],
    #       "\n",
    #       #c[col_to_print],
    #      # "\n",
    #       decoder(c)[col_to_print][0],
    #       "\n",
    #       #d[col_to_print],
    #      # "\n",
    #       decoder(d)[col_to_print][0],
    #       "\n",
    #       #e[col_to_print],
    #      # "\n",
    #       decoder(e)[col_to_print][0],
    #       "\n",
    #       #f[col_to_print],
    #      # "\n",
    #       decoder(f)[col_to_print][0],
    #       "\n",
    #       #g[0],
    #       #"\n",
    #       decoder(g)[col_to_print][0]) #K + 1





    #TEST---------------------------------------------------------------------

    distances = []

    A = tf.transpose(Zs, [1,0,2])
   
    for sf in range(K):

        #B = tf.vectorized_map(_translate_1, A) #2 translations
        
        A = tf.transpose(A, [1,0,2])
        #B = tf.transpose(B, [1,0,2])
        
        valuesA = tf.squeeze(decoder(tf.unstack(A, axis = 0)))
        #valuesB = tf.squeeze(decoder(tf.unstack(B, axis = 0)))
        
        #distances.append(abs(valuesA - valuesB))
        distances.append(valuesA)
        A = tf.transpose(A, [1,0,2])
        
        
        A = tf.vectorized_map(_translate_1, A) 


    #ARGMAX IS ALWAYS WHERE -1, UNLESS OTHER MORE NEGATIVE VALUES ARE ALLOWED
    # (K - (distance to -1 position)) -1 position
    #rotations =  tf.transpose(tf.argmax(distances, axis = 0), [1,0]) + 1
    rotations = (K - 1) -  tf.transpose(tf.argmax(distances, axis = 0), [1,0]) 

    #rotations =  tf.transpose(tf.argmin(distances, axis = 0), [1,0]) + 1

    Ks_predicted = rotations
    
    
    # USE GROUND TRUTH K<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #Ks_predicted = GT_K
    #--------------------------------------------------------------------
    

    Z_ks = []

    for idx, k in enumerate(Ks_predicted): 
   
        aux = []

        for i in range(x.shape[1]):
            a = Zs[i][idx]
            aux.append(LOP.translate_operator_inverse(a, shift=k[i]))
            
        Z_ks.append(aux)
     
    Z_ks = tf.convert_to_tensor(Z_ks, dtype=tf.float32)
    return Z_ks, Ks_predicted



def eval_correctly(x_test, y_test, Zs, Ks, decoder, classifier):
    #"smart" = avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_test, xs)
 
    return classifier.evaluate(x_p, y_test)[1]


# def eval_numeric_rmse(x_dirty, x_clean, Zs, Ks, decoder, headers):
#     #"smart" = avoid decoding Zs when there is no error, just use the value
#     xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
#     x_p = tf.where(Ks == 0, x_dirty, xs)

#     x_df = pd.DataFrame(x_clean, columns = headers["filtered_header"])
#     x_df_p = pd.DataFrame(x_p, columns = headers["filtered_header"])
    
#     return  mean_squared_error(x_df[headers["numeric_header"]], x_df_p[headers["numeric_header"]], squared = False)


def eval_numeric_rmse(x_dirty, Zs, Ks, decoder):
    #"smart" = avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_dirty, xs)

    return x_p
