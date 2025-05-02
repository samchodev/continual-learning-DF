import Model, train 
from Model import *
from train import *
import pandas as pd
import argparse
from tensorflow.keras.optimizers.legacy import Adam

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DFNet with EWC and incremental tasks.")
    parser.add_argument('--first_task', type=int, default=69, help='Number of labels in the first task')
    parser.add_argument('--inc_task', type=int, default=25, help='Number of new labels per incremental task')
    parser.add_argument('--first_epochs', type=int, default=70, help='Epochs for the first task')
    parser.add_argument('--inc_epochs', type=int, default=20, help='Epochs for each incremental task')
    parser.add_argument('--lamb', type=float, default=10000, help='Lambda for EWC regularization')
    args = parser.parse_args()
    
    data = pd.read_pickle('datasets/mon_data.pkl')
    print("data shape :", data.shape)
    MAX_LABEL = 95

    model = DFNet.build(input_shape=(10000, 1), classes=MAX_LABEL)
    OPTIMIZER = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train_loop(
        model, OPTIMIZER, MAX_LABEL, data, test_size=0.2,
        first_task=args.first_task,
        inc_task=args.inc_task,
        first_epochs=args.first_epochs,
        inc_epochs=args.inc_epochs,
        lamb=args.lamb)
