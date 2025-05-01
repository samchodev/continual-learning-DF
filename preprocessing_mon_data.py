import pandas as pd
import numpy as np

def adjust_sequence_length(sequence, target_length=10000, padding_value=-1):
    if isinstance(sequence, (int, float, np.float32, np.float64)):
        sequence = [sequence]
    if len(sequence) < target_length:
        sequence = sequence + [padding_value] * (target_length - len(sequence))
    else:
        sequence = sequence[:target_length]
    return sequence

data = pd.read_csv("mon_standard_dataset.csv")
data = data[['Direction_Size_Sequence', 'Label']]
data['Direction_Sequence'] = data['Direction_Size_Sequence'].apply(lambda x: [1 if i > 0 else -1 for i in eval(x)])
data = data[data['Label'].between(0, 94)]

sequence_array = data['Direction_Sequence'].to_numpy()
sequence_array = np.array([adjust_sequence_length(seq) for seq in sequence_array])

mon_data = pd.DataFrame({
    'Direction_Sequence': list(sequence_array),
    'Label': data['Label'].values 
})

mon_data.to_pickle("mon_data.pkl")
print("Data processed and saved successfully.")