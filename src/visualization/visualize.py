import seaborn as sns
import matplotlib.pyplot as plt

def print_results(cm, accuracy):
    fig = plt.figure(figsize=(8, 6))
    plt.clf()
    res = sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix. Accuracy: {100 * accuracy}%')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()