import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy

def compute_ewc_penalty(model, fisher_matrix, optimal_weights, lamb):   
    loss = 0
    current = model.trainable_weights 
    
    for F, c, o in zip(fisher_matrix, current, optimal_weights):
        o = tf.convert_to_tensor(o, dtype=c.dtype)
        loss += tf.reduce_sum(F * ((c - o) ** 2))


    return loss * (lamb / 2)

def ewc_loss(model, fisher_matrix, lamb, optimal_weights):
    optimal_weights = optimal_weights
    
    def loss_fn(y_true, y_pred):

        ce_loss = CategoricalCrossentropy(from_logits=False)(y_true, y_pred)
        ewc_loss = compute_ewc_penalty(model, fisher_matrix, optimal_weights, lamb=lamb)

        return ce_loss + ewc_loss
    
    return loss_fn

def compute_fisher_matrix(model, data, num_sample=10, epsilon=1e-4):
    epsilon = epsilon
    
    weights = model.trainable_weights
    variance = [tf.zeros_like(tensor) for tensor in weights]

    indices = np.random.choice(len(data), size=num_sample, replace=False)

    for i in indices:

        with tf.GradientTape() as tape:
            tape.watch(weights)
            x = tf.expand_dims(data[i], axis=0)
            output = model(x, training=False) 
            output = tf.clip_by_value(output, epsilon, 1.0)
            log_likelihood = - tf.math.log(output)

        gradients = tape.gradient(log_likelihood, weights)
        for j in range(len(variance)):
                
            if gradients[j] is not None:
                variance[j] += tf.square(gradients[j])

        fisher_matrix = [v / num_sample for v in variance]   

    return fisher_matrix