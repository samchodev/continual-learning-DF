import utils, ewc
from utils import *
from ewc import *

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy


def train_loop(model, OPTIMIZER, MAX_LABEL, data, test_size,
                first_task = 44, inc_task = 5, first_epochs = 30, inc_epochs = 5,
                  lamb=0, num_sample=100):
    
    first_part = split_by_label(data, 0, first_task)
    train, test = split_train_test(first_part, test_size=test_size, random_state=11)
    
    weights = model.trainable_weights
    fisher_matrix = [tf.zeros_like(w) for w in weights]


    i = 1
    while(1):

        if ( first_task + i * inc_task ) <= MAX_LABEL:
            
            if i == 1:
                model.compile(loss=CategoricalCrossentropy(from_logits=False), optimizer=OPTIMIZER, metrics=["accuracy"])
                train_seq, train_label = to_input(train, MAX_LABEL)

                model.fit(x=train_seq, y=train_label, epochs=first_epochs, verbose=2)
                optimal_weights = [w.numpy() for w in model.trainable_weights]

                test_seq, test_label = to_input(test, MAX_LABEL)
                loss, accuracy = model.evaluate(test_seq, test_label, batch_size=32, verbose=1)
                print(f"Task_0 training accuracy: {accuracy:.4f}")


                if len(train_seq) < num_sample:
                    fisher_matrix = compute_fisher_matrix(model, train_seq, num_sample=len(train_seq))
                else:
                    fisher_matrix = compute_fisher_matrix(model, train_seq, num_sample=num_sample)

                i = i + 1
                

            else:

                # training
                model.compile(loss=ewc_loss(model, fisher_matrix, lamb=lamb, optimal_weights=optimal_weights), optimizer=OPTIMIZER, metrics=["accuracy"])

                inc_part = split_by_label(data, first_task + (i-2) * inc_task + 1, first_task + (i-1) * inc_task )
                train, inc_test = split_train_test(inc_part, test_size=test_size, random_state=11)
                train_seq, train_label = to_input(train, MAX_LABEL)

                model.fit(x=train_seq, y=train_label, epochs=inc_epochs, verbose=2)
                optimal_weights = [w.numpy() for w in model.trainable_weights]

                # evaluation
                inc_test_seq, inc_test_label = to_input(inc_test, MAX_LABEL)
                loss, inc_accuracy = model.evaluate(inc_test_seq, inc_test_label, batch_size=32, verbose=1)

                print(f"Task_{i} accuracy: {inc_accuracy:.4f}")

                # update test datset
                test_seq, test_label = to_input(test, MAX_LABEL)

                loss, accuracy = model.evaluate(test_seq, test_label, batch_size=32, verbose=1)
                print(f"Task ~{i-1} accuracy after training on Task_{i}: {accuracy:.4f}")
                
                test = accumulate_data(test, inc_test)

                # calculate fi
                if len(train_seq) < num_sample:
                    fisher_matrix = compute_fisher_matrix(model, train_seq, num_sample=len(train_seq))
                else:
                    fisher_matrix = compute_fisher_matrix(model, train_seq, num_sample=num_sample)

                i = i + 1
            

        else:
            break 
    
    test_seq, test_label = to_input(test, MAX_LABEL)
    loss, accuracy = model.evaluate(test_seq, test_label, batch_size=32, verbose=1)
    print(f"Accuracy after incremental training: {accuracy:.4f}")
