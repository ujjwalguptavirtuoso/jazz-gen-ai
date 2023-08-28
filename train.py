# train.py

import numpy as np
from data_processing import extract_features
from model import build_model

# Load your dataset
# For demonstration purposes, I'll use a placeholder list of file paths.
#file_paths = ["/Users/u0g00bw/Downloads/mode_for_joe/Granted.mp3", "/Users/u0g00bw/Downloads/mode_for_joe/Black.mp3"]
file_paths = ["", ""]  #add the file path

# Define sequence length and prepare data
sequence_length = 35  # This is a hyperparameter, choose based on your data and needs

X = []
y = []

for file in file_paths:
    features = extract_features(file)
    print(f"Features length for {file}: {len(features) if features is not None else 'None'}")

    for i in range(0, len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(features[i + sequence_length])

X_train = np.array(X)
y_train = np.array(y)

# Define the input and output shapes
print(np.shape(X_train))
feature_dimension = X_train.shape[1]
input_shape = (sequence_length, feature_dimension)
output_shape = feature_dimension

model = build_model(input_shape, output_shape)
model.fit(X_train, y_train, epochs=100, batch_size=64)
