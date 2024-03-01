import os
import time
import tensorflow as tf
import numpy as np
from transformation_in_x import apply_transformation_in_x, include_errors_at_random
from tensorflow.keras.layers import Concatenate
import copy 

class LatentTrainLoop(object):

    def __init__(self, autoencoder, encoder, decoder, LOP, epochs, optimizer, train_acc_metric, val_acc_metric, transformation):#, tracker):

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.LOP = LOP
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_acc_metric = train_acc_metric
        self.val_acc_metric = val_acc_metric
        self.transformation = transformation
        self.train_acc = 0.0
        self.val_acc = 0.0
        self.concatenation = Concatenate()
        #self.num_columns = columns
        self.loss_function = tf.keras.losses.MeanSquaredError()





    
    @tf.function
    def translate_operator_both_ways_indexed(self, inputs):
        zs , Ks_differences = inputs
        column = 0

        
        for k in Ks_differences:
            if k > 0:
                new_z = self.LOP.translate_operator_inverse(zs[column], abs(k))
            else:
                new_z = self.LOP.translate_operator(zs[column], abs(k))

            zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
            column = column + 1

        return zs



    # @tf.function
    # def translate_operator_indexed(self, inputs):
    #     zs , KS = inputs
    #     column = 0
    #     for k in KS:
    #         new_z = self.LOP.translate_operator(zs[column], k)
    #         #new_z = self.LOP.translate_operator_1D(zs[column], k)
    #         zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
    #         column = column + 1
    #     return zs

    
    # @tf.function
    # def translate_operator_inverse_indexed(self, inputs):
    #     zs , KS = inputs
    #     column = 0
    #     for k in KS:
    #         new_z = self.LOP.translate_operator_inverse(zs[column], k)
    #         #new_z = self.LOP.translate_operator_inverse_1D(zs[column], k)
    #         zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
    #         column = column + 1
    #     return zs

        

    @tf.function
    def custom_loss(self, y, y_rec, y_transformed, y_transformed_rec, z,z_p):

        #x1_reconstruction_loss = tf.reduce_mean(tf.square(y - y_rec))
        #x2_reconstruction_loss = tf.reduce_mean(tf.square(y_transformed - y_transformed_rec))

        #x1_reconstruction_loss = self.loss_function(y, y_rec)
        #x2_reconstruction_loss = self.loss_function(y_transformed, y_transformed_rec)
        B_reconstruction_loss = self.loss_function(z, z_p)

        #return x1_reconstruction_loss + x2_reconstruction_loss + B_reconstruction_loss
        #return x1_reconstruction_loss + B_reconstruction_loss
        return B_reconstruction_loss
        

    
    @tf.function
    def _trans(self,z1,z2,z3):
        return tf.transpose(z1,[1,0,2]), tf.transpose(z2,[1,0,2]), tf.transpose(z3,[1,0,2])
    
    @tf.function
    def _decode(self,Z):
        return self.concatenation(self.decoder(tf.unstack(Z, axis = 0)))
        

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:

            K = self.LOP.n_rotations
            K2 = self.LOP.n_left_shifts
            T = self.transformation

            #SHIFT BATCH-WISE====================================
            Z_vectors =  self.encoder(x)

            #TODO: PRECALCULATE ALL TRANSFORMED X TO BE USED, NOT AS IMPACTFUL AS HAVING THE HUGE MATRIX
            x_transformed, missing = include_errors_at_random(x, K, 2.0)
            Zs_transformed =  self.encoder(x_transformed)            
            Zs_B = self.encoder(x_transformed)

            
            #new idea
            x_transformed_B, missing_B = include_errors_at_random(x, K, 2.0)
            
            

            #TODO: TRY WITH 1 D TRANSLATION TO MAKE THINGS EASIER AND FASTER

            start = time.time()


            Zs_transformed, Z_vectors, Zs_B = self._trans(Zs_transformed, Z_vectors, Zs_B)



            #TODO: SINGLE MATRIX TO MULTIPLY ALL COLUMNS, NOT COLUMN BY COLUMN <SLOWER
            #Zs_transformed = tf.vectorized_map(self.translate_operator_inverse_indexed, [Zs_transformed, missing])  
            #Z_vectors = tf.vectorized_map(self.translate_operator_indexed, [Z_vectors, missing])


            

            #Zs_B = tf.vectorized_map(self.translate_operator_inverse_indexed, [Zs_B, missing])
            #Zs_B = tf.vectorized_map(self.translate_operator_indexed, [Zs_B, missing_B]) 

            #TODO: could move directly to the missing_B position, 1 second per MAP operations
            diffs = missing - missing_B

            # tf.print(diffs.shape, Zs_B.shape)

            #Zs_B = tf.map_fn(self.translate_operator_both_ways_indexed, [Zs_B, diffs]) #tf.vectorized_map(self.translate_operator_both_ways_indexed, [Zs_B, diffs])
            Zs_B = tf.vectorized_map(self.translate_operator_both_ways_indexed, [Zs_B, diffs])
            
            
            Zs_transformed, Z_vectors, Zs_B = self._trans(Zs_transformed, Z_vectors, Zs_B)
            
            
            #logits_x  = self.concatenation(self.decoder(tf.unstack(Zs_transformed, axis = 0)))
            #logits_xt  =  self.concatenation(self.decoder(tf.unstack(Z_vectors, axis = 0)))
            #logits_B  =  self.concatenation(self.decoder(tf.unstack(Zs_B, axis = 0)))

            #logits_x  = self._decode(Zs_transformed)
            #logits_xt  =  self._decode(Z_vectors)
            logits_B  =  self._decode(Zs_B)
            
            #tf.print(time.time() - start)
            

            # loss_value = self.custom_loss(y, logits_x,
            #                               x_transformed, logits_xt,
            #                               x_transformed_B, logits_B)

            loss_value = self.custom_loss(0, 0,
                                          0, 0,
                                          x_transformed_B, logits_B)


            start = time.time()
            grads = tape.gradient(loss_value, self.autoencoder.trainable_weights)
            #tf.print("grads: ", time.time() - start)

        start = time.time()
        self.optimizer.apply_gradients(zip(grads,  self.autoencoder.trainable_weights))
        #tf.print("apply grads: ", time.time() - start)
        
        #self.train_acc_metric.update_state(y, logits_x)
        self.train_acc_metric.update_state(x_transformed_B, logits_B)
    
        return loss_value














    


    @tf.function
    def test_step(self, x, y):
        outputs = self.autoencoder(x, training=False)
        val_logits = outputs[0]
        #tf.print(y[0],"\n", val_logits[0])
        self.val_acc_metric.update_state(y, val_logits)



    #@tf.function
    def train_loop(self, train_dataset, val_dataset):
        for epoch in range(self.epochs):
            
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                try:                
                    loss_value = self.train_step(x_batch_train, y_batch_train)
                #TODO: randomly trowing this exception during training due to size mismatch!!
                except tf.errors.InvalidArgumentError: 
                    pass


            # Display and reset metrics at the end of each epoch.
            self.train_acc = self.train_acc_metric.result()
            self.train_acc_metric.reset_states()

            
            #VALIDATE
            if epoch % 10 == 0:
                for x_batch_val, y_batch_val in val_dataset:
                    self.test_step(x_batch_val, y_batch_val)

                self.val_acc = self.val_acc_metric.result()
                tf.print(self.train_acc,"    ", self.val_acc)
                self.val_acc_metric.reset_states()
                
            


        for x_batch_val, y_batch_val in val_dataset:
            self.test_step(x_batch_val, y_batch_val)

        self.val_acc = self.val_acc_metric.result()
        self.val_acc_metric.reset_states()






####################################################################################################
####################################################################################################
####################################################################################################


# class UnshifterLatentTrainLoop(LatentTrainLoop):

#     def __init__(self, autoencoder, encoder, decoder, LOP, epochs, optimizer, train_acc_metric, val_acc_metric, transformation):

#         self.autoencoder = autoencoder
#         self.encoder = encoder
#         self.decoder = decoder
#         self.LOP = LOP
#         self.epochs = epochs
#         self.optimizer = optimizer
#         self.train_acc_metric = train_acc_metric
#         self.val_acc_metric = val_acc_metric
#         self.transformation = transformation
#         self.train_acc = 0.0
#         self.val_acc = 0.0
#         self.concatenation = Concatenate()
#         #self.num_columns = columns
#         self.loss_function = tf.keras.losses.MeanSquaredError()

#         for el in self.encoder.layers:
#             el.trainable = False


#     @tf.function
#     def train_step(self, x, y):
#         with tf.GradientTape() as tape:

#             K = self.LOP.n_rotations
#             K2 = self.LOP.n_left_shifts
#             T = self.transformation

#             Z_vectors =  self.encoder(x)

#             x_transformed, missing = include_errors_at_random(x, K, 3.0)
#             Zs_transformed =  self.encoder(x_transformed)


#             #Zs_transformed = tf.transpose(Zs_transformed, [1,0,2])
#             #Z_vectors = tf.transpose(Z_vectors, [1,0,2])

#             #Zs_transformed = tf.vectorized_map(self.translate_operator_inverse_indexed, [Zs_transformed, missing])  
#             #Z_vectors = tf.vectorized_map(self.translate_operator_indexed, [Z_vectors, missing]) 
            
#             #Zs_transformed = tf.transpose(Zs_transformed, [1,0,2])
#             #Z_vectors = tf.transpose(Z_vectors, [1,0,2])
            
#             logits_x  = self.concatenation(self.decoder(tf.unstack(Zs_transformed, axis = 0)))
#             logits_xt  =  self.concatenation(self.decoder(tf.unstack(Z_vectors, axis = 0)))

#             loss_value = self.custom_loss(y, logits_x,
#                                           y, logits_xt, #<<<<<<<<<<<<<<<<<<<<<< BOTH GO TO THE CLEAN EXAMPLE
#                                           Zs_transformed, Z_vectors)

#             start = time.time()
#             grads = tape.gradient(loss_value, self.autoencoder.trainable_weights)


#         start = time.time()
#         self.optimizer.apply_gradients(zip(grads,  self.autoencoder.trainable_weights))
#         #tf.print("apply grads: ", time.time() - start)
        
#         self.train_acc_metric.update_state(y, logits_x)
    
#         return loss_value






    
#     #@tf.function
#     def train_step(self, x, y):
#         with tf.GradientTape() as tape:

#             K = self.LOP.n_rotations
#             K2 = self.LOP.n_left_shifts
#             T = self.transformation

#             #SHIFT BATCH-WISE====================================
#             Z_vectors =  self.encoder(x)

#             #TODO: PRECALCULATE ALL TRANSFORMED X TO BE USED, NOT AS IMPACTFUL AS HAVING THE HUGE MATRIX
#             x_transformed, missing = include_errors_at_random(x, K, 3.0)
#             Zs_transformed =  self.encoder(x_transformed)


#             #new idea
#             x_transformed_B, missing_B = include_errors_at_random(x, K, 3.0)
#             Zs_B = self.encoder(x_transformed)
            

#            #TODO: ZERO IS TRIPLE DIPPING!!!


#             Zs_transformed = tf.transpose(Zs_transformed, [1,0,2])
#             Z_vectors = tf.transpose(Z_vectors, [1,0,2])
#             Zs_B = tf.transpose(Zs_B, [1,0,2])


#             Zs_transformed = tf.vectorized_map(self.LOP.translate_operator_all_columns, Zs_transformed)
#             Z_vectors = tf.vectorized_map(self.LOP.translate_operator_all_columns_inverse, Z_vectors)  
           
#             start = time.time()
#             all_transformed = []
#             all_vectors = []
#             for idx in range(x.shape[0]):
#                 all_transformed.append(self.LOP.translate_operator_all_columns_inverse(Zs_transformed[idx]))
#                 all_vectors.append(self.LOP.translate_operator_all_columns(Z_vectors[idx]))
                

#             #all_transformed = tf.vectorized_map(self.LOP.translate_operator_all_columns_inverse, Zs_transformed)
#             #all_vectors = tf.vectorized_map(self.LOP.translate_operator_all_columns, Z_vectors)

#             tf.print(time.time() - start)


#             z_dim = 60
#             tf.print(len(all_transformed), len(all_transformed[0]))
#             tf.print(self.LOP.get_latent_reshaped_for(2, 3, all_transformed, z_dim).shape)

            
#             start = time.time()
#             Zs_transformed = []

#             tuple_idx = 0
            
#             for m in missing:
# #                aux = []
#  #               for j in range(self.LOP.n_columns):
#   #                  aux.append(self.LOP.get_latent_reshaped_for(m[j], j, all_transformed[:][tuple_idx], z_dim))
#    #             Zs_transformed.append(aux)
#                 Zs_transformed.append(self.LOP.get_latent_reshaped_for_all_columns(m, all_transformed[:][tuple_idx], z_dim))
#                 tuple_idx += 1
#             Zs_transformed = tf.transpose(Zs_transformed, [1,0,2])
#             tf.print(time.time() - start)

#             tf.print(Zs_transformed.shape)

#             tuple_idx = 0
            
#             start = time.time()
#             Z_vectors = []
#             for m in missing:
#                 #aux = []
#                 #for j in range(self.LOP.n_columns):
#                 Z_vectors.append(self.LOP.get_latent_reshaped_for_all_columns(m, all_vectors[:][tuple_idx], z_dim))
#                 #Z_vectors.append(aux)
#                 tuple_idx += 1
#             Z_vectors = tf.transpose(Z_vectors, [1,0,2])
#             tf.print(time.time() - start)

#             tf.print(Z_vectors.shape)


                
#             #new idea
#             #Zs_B = tf.vectorized_map(self.translate_operator_inverse_indexed, [Zs_B, missing])
#             #Zs_B = tf.vectorized_map(self.translate_operator_indexed, [Zs_B, missing_B])
            
#             #Zs_B = tf.vectorized_map(LOP.translate_operator_all_columns_inverse, Zs_B)
#             Zs_B = Z_vectors 

                



#             #Zs_transformed = tf.transpose(Zs_transformed, [1,0,2])
#             #Z_vectors = tf.transpose(Z_vectors, [1,0,2])
#             #Zs_B = tf.transpose(Zs_B, [1,0,2])



#             start = time.time()
#             logits_x  = self.concatenation(self.decoder(tf.unstack(Zs_transformed, axis = 0)))
#             logits_xt  =  self.concatenation(self.decoder(tf.unstack(Z_vectors, axis = 0)))
#             logits_B  =  self.concatenation(self.decoder(tf.unstack(Zs_B, axis = 0)))
#             tf.print(time.time() - start)

            
#             loss_value = self.custom_loss(y, logits_x,
#                                           x_transformed, logits_xt,
#                                           x_transformed_B, logits_B)

#             start = time.time()
#             grads = tape.gradient(loss_value, self.autoencoder.trainable_weights)
#             #tf.print("grads: ", time.time() - start)

#         start = time.time()
#         self.optimizer.apply_gradients(zip(grads,  self.autoencoder.trainable_weights))
#         #tf.print("apply grads: ", time.time() - start)
        
#         self.train_acc_metric.update_state(y, logits_x)
    
#         return loss_value

