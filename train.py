import torch.nn as nn
import torchvision.models as models

import warnings
warnings.filterwarnings('ignore')
import os
import time

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

#Updating with verbose tqdm train and test functions
from tqdm import tqdm  # For Jupyter-specific progress bar
import logging
import time

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    ## Set Hyperparameters
class Params:
    def __init__(self):
        self.batch_size = 32
        self.name = "resnet_50_sgd1"
        self.workers = 4
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# Configure logging for Jupyter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()

    # Use tqdm for progress visualization
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = len(X)
        step = epoch * size + (batch + 1) * batch_size

        # Update tqdm description and writer
        if batch % 100 == 0:
            current_loss = loss.item()
            progress_bar.set_postfix({"loss": current_loss, "progress": f"{(batch+1)*batch_size}/{size}"})
            if writer is not None:
                writer.add_scalar('training loss', current_loss, step)
            logger.info(f"Batch {batch+1}: loss={current_loss:.6f}, progress={(batch+1)*batch_size}/{size}")

    epoch_time = time.time() - start0
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")


def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, calc_acc5=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0

    # Use tqdm for progress visualization
    progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}")

    with torch.no_grad():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size
    top5_accuracy = 100 * correct_top5 / size if calc_acc5 else None

    step = epoch * len(train_dataloader.dataset)
    if writer is not None:
        writer.add_scalar('test loss', test_loss, step)
        writer.add_scalar('test accuracy', accuracy, step)
        if calc_acc5:
            writer.add_scalar('test accuracy5', top5_accuracy, step)

    logger.info(f"Test Results - Epoch {epoch+1}: Accuracy={accuracy:.2f}%, Avg loss={test_loss:.6f}")
    if calc_acc5:
        logger.info(f"Top-5 Accuracy={top5_accuracy:.2f}%")

if __name__ == "__main__":
    params = Params()
    print(params, params.batch_size)

    training_folder_name = '/content/imagenetmini-1000/imagenet-mini/train'
    val_folder_name = '/content/imagenetmini-1000/imagenet-mini/val'

    train_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            # Normalize the pixel values (in R, G, and B channels)
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers = params.workers,
        pin_memory=True,
    )

    val_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=256, antialias=True),
            transforms.CenterCrop(224),
            # Normalize the pixel values (in R, G, and B channels)
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name,
        transform=val_transformation
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=params.workers,
        shuffle=False,
        pin_memory=True
    )

    # device
    print("Libraries imported - ready to use PyTorch", torch.__version__)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    ## Testing with pre-trained model : only to be done once
    ## testing a pretrained model to validate correctness of our dataset, transform and metrics code
    # pretrained_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT').to(device)
    # start = time.time()
    # loss_fn = nn.CrossEntropyLoss()
    # test(val_loader, pretrained_model, loss_fn, epoch=0, writer=None, train_dataloader=train_loader, calc_acc5=True)
    # print("Elapsed: ", time.time() - start)

    # resume training options
    resume_training = True

    num_classes = len(train_dataset.classes)
    model = ResNet50(num_classes=num_classes)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)

    ## Current State of Training
    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    print(checkpoint_path)
    if resume_training and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        assert params == checkpoint["params"]

    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter('runs/' + params.name)
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, calc_acc5=True)
    print("Starting training")
    for epoch in range(start_epoch, 10):   
        print(f"Epoch {epoch}")
        train(train_loader, model, loss_fn, optimizer, epoch=epoch, writer=writer)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))
        lr_scheduler.step()
        test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader, calc_acc5=True)