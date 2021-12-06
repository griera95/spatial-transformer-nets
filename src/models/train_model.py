import torch
import torch.nn.functional as F
import torch.optim as optim


def train(cfg, model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=cfg.model.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(1, cfg.model.epochs + 1):
        train_step(model, train_loader, optimizer, device, i)
    


def train_step(model, train_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))