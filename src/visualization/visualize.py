import seaborn as sns
import matplotlib.pyplot as plt

def print_results(cm, accuracy, test_loss):
    fig = plt.figure(figsize=(8, 6))
    plt.clf()
    res = sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix. Accuracy: {100 * accuracy}%, test loss: {test_loss:.6f}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()

def print_training_evolution(losses):
    fig = plt.figure(figsize=(8, 6))
    plt.clf()
    plt.title(f'Training loss evolution')
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.plot(losses)
    plt.savefig('training_evolution.png')
    plt.show()
    plt.close()