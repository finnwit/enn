import matplotlib.pyplot as plt
import numpy as np

# Exercise 1 - plotting the training curve
def plot_training_curve(loss_history, path):
    """
    Plot training loss over epochs and save to file.

    Parameters
    ----------
    loss_history : list or np.ndarray
        Loss values collected during training.
    path : str
        Directory and file name for the output PDF file.

    Returns
    -------
    str
        Path to the saved figure.
    """
    plt.figure(figsize=(6, 4))
    # TODO: Implement a plot function that visualizes the Learning over time (as given in the loss_history)

    return path

# Exercise 1 - Visualization of the data set and decision regions
def plot_decision_regions(model, X, y, path):
    """
    Plot decision regions and training data points

    Parameters
    ----------
    model : 
        fitted model, can be used to predict any point in the plane
    X:
        training inputs
    y:
        training targets
    path : str
        Directory and file name for the output PDF file.

    Returns
    -------
    str
        Path to the saved figure.
    """
    plt.figure(figsize=(6, 6))
    # TODO: You have to visualize the decision regions of the given model.
    # Approach: 
    # - Create a lot of points in an area
    #   (in a single dimension using for example np.linspace,
    #    and for the 2D putting this in a np.meshgrid).
    # - predict the class for all of these using model.predict
    # - visualize this, e.g. using contourf plots 

    # The data points could - for example - be shown in a scatter plot:
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title("Linear decision regions")

    plt.savefig(path)
    plt.close()

    return path

# Exercise 3 - Visualize accuracies depending on the hidden layer size
# at the end of training
def plot_hidden_size_vs_accuracy(
    hidden_sizes,
    train_accuracies,
    test_accuracies,
    path
):
    """
    Plot training and test accuracy as a function of hidden layer size.

    Parameters
    ----------
    hidden_sizes : list[int]
        Number of neurons in the hidden layer.
    train_accuracies : list[float]
        Training accuracies for each hidden size.
    test_accuracies : list[float]
        Test accuracies for each hidden size.
    path : str
        Directory and file name of the output PDF file.

    Returns
    -------
    str
        Path to the saved PDF file.
    """
    # TODO: create a plot that visualizes the training
    # and test accuracies (given on the y-axis)
    # with respect to the hidden layer size.

    return path