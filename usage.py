import numpy as np
import functions as fn

# Example: Student passing grades prediction
# Features: [bias, math_score, english_score]
# Label: 1 = pass, 0 = fail
# Rescale function for feature columns (excluding bias)
def rescale_matrix(X):
    X = np.array(X, dtype=float)
    X_scaled = X.copy()
    # Exclude bias (first column)
    for col in range(1, X.shape[1]):
        min_val = X[:, col].min()
        max_val = X[:, col].max()
        if max_val > min_val:
            X_scaled[:, col] = (X[:, col] - min_val) / (max_val - min_val)
        else:
            X_scaled[:, col] = 0.0
    return X_scaled

# You can now modify these arrays as you wish, then call rescale_matrix()
toy_X_train = np.array([
    [1.0, 40, 60],   # fail (low math)
    [1.0, 55, 45],   # fail (low math)
    [1.0, 65, 50],   # pass (math above 60)
    [1.0, 80, 30],   # pass (high math, low english)
    [1.0, 50, 80],   # fail (math not enough)
    [1.0, 90, 90],   # pass (high both)
])
toy_X_train = rescale_matrix(toy_X_train)
toy_Y_train = np.array([[0], [0], [1], [1], [0], [1]])
toy_X_val = np.array([
    [1.0, 60, 40],   # pass (math just enough)
    [1.0, 45, 85],   # fail (math too low)
])
toy_X_val = rescale_matrix(toy_X_val)
toy_Y_val = np.array([[1], [0]])
toy_X_test = np.array([
    [1.0, 70, 60],   # pass
    [1.0, 55, 90],   # fail
    [1.0, 85, 40],   # pass
    [1.0, 50, 50],   # fail
    [1.0, 75, 85],   # pass
])
toy_X_test = rescale_matrix(toy_X_test)
num_epochs = 1000
lr = 0.05

# training and validation
output = fn.train_and_val(X_train=toy_X_train, 
                       Y_train=toy_Y_train, 
                       X_val=toy_X_val, 
                       Y_val=toy_Y_val,
                       num_epochs=num_epochs, 
                       lr=lr)

model = output['best_theta']

predictions = fn.predict(toy_X_test, model)
print(predictions)