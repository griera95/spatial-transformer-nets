import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

# setup logger
log = logging.getLogger('__main__')
log.setLevel(logging.INFO)

def test(model, test_loader):
    """Evaluate model performance returning confusion matrix, loss and accuracy on test set"""
    # select cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # set model to eval mode to not track gradients
        model.eval()

        # initialize test loss and correct count to 0 before evaluating on batches
        test_loss = 0
        correct = 0
        # initialize empty list for storing confusion matrices
        cms = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # compute model output
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            # calculate conf matrix and append
            batch_cm = confusion_matrix(torch.flatten(target), torch.flatten(pred), labels=list(range(0, 10)))
            cms.append(batch_cm)
            # update correct count
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        # average test loss and accuracy over number of examples
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset), 100. * accuracy))

        # summ all partial confusion matrices
        cm = np.zeros((10, 10), int)
        for m in cms:
            cm = cm + m
        cm

    return cm, accuracy, test_loss