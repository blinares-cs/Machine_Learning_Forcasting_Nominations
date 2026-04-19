import numpy as np
import argparse
import matplotlib.pyplot as plt


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    """
    Training function.

    Parameters:
        X (np.ndarray): Input features, np.ndarray.
        y (np.ndarray): Labels, np.ndarray
        num_epoch : Number of epochs, int
        learning_rate: Learning rate, float

    Returns:
        X augmented (with intercept) np.ndarray and theta augmented, np.ndarray
    """
    N, D = X.shape
    # incorporate intercept into X by adding a column of ones
    X_aug = np.hstack([X, np.ones((N, 1))])  # shape (N, D+1)
    # initialize theta_aug including intercept (last element)
    theta_aug = np.zeros(D + 1, dtype=np.float64)

    losses = []   # store NLL per epoch

    for epoch in range(num_epoch):
        for i in range(N):
            X_aug_i = X_aug[i]
            y_i = y[i]
            # gradient
            grad = X_aug_i.T * ((sigmoid(X_aug_i @ theta_aug)) - y_i)    # shape (D+1,)
            # theta
            theta_aug -= learning_rate * grad

    
        # compute average NLL after epoch 
        probs = sigmoid(X_aug @ theta_aug)
        nll = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        losses.append(nll)


    # print(" -- finished training -- ")
    return X_aug, theta_aug, losses


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    """
    Predicting 0 or 1 values for features

    Parameters:
        X (np.ndarray): Input features, np.ndarray.
        theta (np.ndarray): Weights, np.ndarray

    Returns:
        Array with predictions (list of 0s and 1s)
    """
    predictions = []
    theta_extract = theta[:-1]  # feature weights
    c_extract = theta[-1]       # intercept
    for i in range(len(X)):
        z = np.dot(theta_extract, X[i]) + c_extract
        y_pred = sigmoid(z)
        # predictions.append(y_pred)
        if y_pred>=0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    """
    Compute error (compare predicted and true values)

    Parameters:
        y_pred (np.ndarray): Predicted labels from logistic regression, np.ndarray.
        y (np.ndarray): True labels, np.ndarray

    Returns:
        Error of predictions (float)
    """
    correct_pred = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y[i]:
            correct_pred +=1
    error = 1-(correct_pred/len(y_pred))
    return f"{error:.6f}"


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    # ----- TRAINING RESULTS -----
    # X is features (everything except for first col) and y is labels (first col)
    X = []
    y = []
    with open(args.train_input) as f:
        data = (line.strip().split('\t') for line in f)
        for row in data:
            X.append(row[1:]) # Skip first element (labels) and include the rest (features)
            y.append(row[0]) # Keep only first col (labels)
    # Convert to np.array of type float
    X = np.array(X)
    y = np.array(y)
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    theta = np.zeros_like(X[0]) # weight array with shape equal to num features
    num_epoch = args.num_epoch
    learning_rate = args.learning_rate
    X_aug, theta_aug, train_losses = train(theta, X, y, num_epoch, learning_rate)

    X = X_aug[:, :-1]  # preserve everything except last col (which stores intercept value)

    # training predictions
    predictions = predict(theta_aug, X)

    with open(args.train_out, "w") as file:
        for pred in predictions:
            file.write(str(pred) + '\n')

    # ----- TESTING RESULTS -----
    X_test = []  
    y_test = []
    with open(args.test_input) as f:
        data = (line.strip().split('\t') for line in f)
        for row in data:
            X_test.append(row[1:])
            y_test.append(row[0])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # make predictions using the trained model from earlier
    predictions_test = predict(theta_aug, X_test)

    with open(args.test_out, "w") as file:
        for pred in predictions_test:
            file.write(str(pred) + '\n')

    train_error = compute_error(predictions,y)
    # print("training error: " + str(train_error))
    test_error = compute_error(predictions_test,y_test)
    # print("testing error: " + str(test_error))
    with open(args.metrics_out, "w") as file:
        file.write("error(train): " + train_error + '\n')
        file.write("error(test): " + test_error + '\n')
