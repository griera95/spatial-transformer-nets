import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

log = logging.getLogger('__main__')
log.setLevel(logging.INFO)

def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        cms = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            cms.append(confusion_matrix(torch.flatten(target), torch.flatten(pred), labels=list(range(0, 10))))
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset), 100. * accuracy))

        cm = np.zeros((10, 10), int)
        for m in cms:
            cm = cm + m
        cm

    return cm, accuracy, test_loss