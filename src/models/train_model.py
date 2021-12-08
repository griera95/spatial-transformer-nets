import torch
import torch.nn.functional as F
import torch.optim as optim
import logging

# setup logging
log = logging.getLogger('__main__')
log.setLevel(logging.INFO)


def train(cfg, model, train_loader):
    """Train model on training set for a number of epochs defined in config"""
    # get optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.model.lr)

    # select cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize empty array for storing epoch losses
    losses = []

    # call training_step for a number of epochs specified in config
    for i in range(1, cfg.model.epochs + 1):
        losses.append(train_step(model, train_loader, optimizer, device, i))

    return losses
    


def train_step(model, train_loader, optimizer, device, epoch):
    """Perform one training epoch"""
    # Initialize step_loss to 0 for accumulating
    step_loss = 0.
    # set model to training mode for keeping track of gradients
    model.train()
    # iterate over batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the gradients from previous steps
        optimizer.zero_grad()
        # compute model output
        output = model(data)
        # loss of the model output w.r.t. the targets
        loss = F.nll_loss(output, target)
        # propagate loss backwards
        loss.backward()
        # update model parameters according to optimizer
        optimizer.step()
        if batch_idx % 500 == 0:
            # print progress every 500 batches
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        # update loss
        step_loss += loss.item()
    
    # average loss over number of batches
    step_loss /= (batch_idx+1)

    return step_loss