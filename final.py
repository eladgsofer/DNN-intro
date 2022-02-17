from torch.utils.data.dataloader import DataLoader

from torch.utils.data import random_split
from models import ResNet
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from utills import *
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


random_seed = 42
torch.manual_seed(random_seed)


class GenericClassifier():
    def __init__(self, model):
        self.model = model

    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def evaluate(self, model, val_loader):
        outputs = []
        # Notify pytorch eval mode
        model.eval()
        with torch.no_grad:
            for batch in val_loader:

                images, labels = batch
                # Generate predictions
                out = self.model.forward(images)
                # Calculate loss
                loss = F.cross_entropy(out, labels)
                # Calculate accuracy
                acc = self.accuracy(out, labels)
                outputs.append({'val_loss': loss.detach(), 'val_acc': acc})

            batch_losses = map(lambda x: x['val_loss'], outputs)
            batch_accs = map(lambda x: x['val_acc'], outputs)

            # Aggregate and average the loss & accuracy
            epoch_loss = torch.stack(batch_losses).mean()
            epoch_acc = torch.stack(batch_accs).mean()
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def fit(self, epochs, lr, model, train_loader, val_loader,opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_losses = []
            for batch in train_loader:
                images, labels = batch
                # TODO ENCODER

                # Generate predictions
                out = self.model.forward(images)
                loss = F.cross_entropy(out, labels)  # Calculate loss
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation phase
            result = self.evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()

            print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))
            history.append(result)
        return history

    def show_batch(self, dl):
        images, labels = next(dl)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))



if __name__ == '__main__':
    num_epochs = 8
    opt_func = torch.optim.Adam
    lr = 5.5e-5
    batch_size = 32

    data_dir = '/Users/elad.sofer/src/DeepLearning/garbage/Garbage classification/Garbage classification'

    ################################################
    # Transformers
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # Load & Split the Data
    dataset = ImageFolder(data_dir, transform=transformations)
    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size * 2, pin_memory=True)
    # Move to GPU
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    ################################################

    model = ResNet(dataset)
    to_device(model, device)
    classifier = GenericClassifier(model)

    history = classifier.fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    plot_accuracies(history)
    plot_losses(history)





