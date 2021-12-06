import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def print_results(cm, accuracy, test_loss):
    """Print confusion matrix, accuracy and loss"""

    # define figure size
    fig = plt.figure(figsize=(8, 6))

    # clear previous images
    plt.clf()

    # generate conf matrix with seaborn
    res = sns.heatmap(cm, annot=True, fmt='d')

    # annotate
    plt.title(f'Confusion Matrix. Accuracy: {100 * accuracy}%, test loss: {test_loss:.6f}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # save image to file
    plt.savefig('confusion_matrix.png')

    #show image
    plt.show()

    plt.close()

def print_training_evolution(losses):
    """Print training evolution"""

    # define figure size
    fig = plt.figure(figsize=(8, 6))

    # clear previous images
    plt.clf()

    # annotate
    plt.title(f'Training loss evolution')
    plt.xlabel('epoch')
    plt.ylabel('log(loss)')

    # plot logarithm of the losses
    plt.plot(np.log(losses))

    # save image to file
    plt.savefig('training_evolution.png')

    #show image
    plt.show()
    plt.close()