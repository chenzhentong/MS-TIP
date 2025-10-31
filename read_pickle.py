import pickle

# Specify the path to your .pkl file
#file_path = '/home/czt/project/open-source-projects/MS-TIP/checkpoint/end_point_eth_experiment/constant_metrics-0.15.pkl'

file_path='/home/czt/project/open-source-projects/MS-TIP/datasets/eth/test/biwi_eth.pkl'
# Open the file in binary mode
with open(file_path, 'rb') as file:
    # Load the content of the Pickle file
    data = pickle.load(file)

# Print the content
print(data)
