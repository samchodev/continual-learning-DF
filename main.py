import Model, train 
from Model import *
from train import *
import pandas as pd
from tensorflow.keras.optimizers.legacy import Adam

if __name__ == "__main__":

    data = pd.read_pickle('mon_data.pkl')
    print("data shape :", data.shape)
    MAX_LABEL = 94

    model = DFNet.build(input_shape=(10000, 1), classes=MAX_LABEL)
    OPTIMIZER = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train_loop(model, OPTIMIZER, MAX_LABEL, data, test_size=0.2, first_task = 49, inc_task = 45, first_epochs = 80, inc_epochs = 20, lamb=0, num_sample=100)