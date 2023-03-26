import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

TRAIN ='train'
VALIDATION = 'val'

MAX_SAMPLES_PROPORTION = 0.2
THESHOLD_LOGGING_SAMPLES = 2000

def log_metrics(loss, running_corrects, running_samples, total_samples):
    accuracy = running_corrects / running_samples
    print("Images [{}/{} ({:.0f}%)] Loss: {:.3f} Accuracy: {}/{} ({:.3f}%)".format(
        running_samples,
        total_samples,
        100.0 * (running_samples / total_samples),
        loss.item(),
        running_corrects,
        running_samples,
        100.0 * accuracy,
        )
    )


def validate(model, validation_loader, criterion):
    model.eval()
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    total_samples = len(validation_loader.dataset)

    for inputs, labels in validation_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)
        if running_samples % THESHOLD_LOGGING_SAMPLES == 0:
            log_metrics(loss, running_corrects, running_samples, total_samples)
            
        #NOTE: Comment lines below to train and test on whole dataset
        if running_samples > (MAX_SAMPLES_PROPORTION * total_samples):
            break

    epoch_loss = running_loss / running_samples
    epoch_acc = running_corrects / running_samples
    print(f"Phase validation, Epoc loss {epoch_loss:.3f}, Epoc accuracy {100*epoch_acc:.3f}%")
    return epoch_loss


def train(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    total_samples = len(train_loader.dataset)

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)
        if running_samples % THESHOLD_LOGGING_SAMPLES == 0:
            log_metrics(loss, running_corrects, running_samples, total_samples)

        #NOTE: Comment lines below to train and test on whole dataset
        if running_samples > (MAX_SAMPLES_PROPORTION * total_samples ):
            break

    epoch_loss = running_loss / running_samples
    epoch_acc = running_corrects / running_samples
    print(f"Phase training, Epoc loss {epoch_loss:.3f}, Epoc accuracy {100*epoch_acc:.3f}%")
    return epoch_loss

def train_with_early_stopping(model, datasets_loader, epochs, loss_criterion, optimizer):
    print(f"Training Model on {MAX_SAMPLES_PROPORTION*100}% of the Dataset")
    best_loss = 1e6
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch} ...")
        _ = train(model, datasets_loader[TRAIN], loss_criterion, optimizer)
        validate_epoch_loss = validate(model, datasets_loader[VALIDATION], loss_criterion)
        print(validate_epoch_loss, best_loss)
        if validate_epoch_loss < best_loss:
            best_loss = validate_epoch_loss
        else:
            print('Loss of validation model started to increase')
            break
            
def net(num_classes: int):
    '''Initializes a pretrained model'''
    model = models.resnet50(pretrained=True)

    # Freeze training of the convolutional layers
    for param in model.parameters():
        param.requires_grad = False   

    # Override the last layer to adjust it to our problem
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes)
    )
    
    return model

def create_data_loaders(train_data_dir: str, valid_data_dir: str, batch_size: int):
    '''Create pytorch data loaders'''
     
    data_dir = {TRAIN: train_data_dir, VALIDATION: valid_data_dir}
    
    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        VALIDATION: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
   
    image_datasets = {}
    dataloaders = {}
    for x in [TRAIN, VALIDATION]:
        
        image_datasets[x] = torchvision.datasets.ImageFolder(data_dir[x], data_transforms[x])
        dataloaders[x] = torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
        
    return dataloaders

def get_num_classes(dataloader) -> int:
        return len(dataloader[TRAIN].dataset.classes)
    
def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    print(f"Saving the model to path {path}")
    torch.save(model.state_dict(), path)
    
def main(args):
    
    dataset_loaders = create_data_loaders(args.data_dir_train, args.data_dir_validation, args.batch_size)
    
    num_classes = get_num_classes(dataset_loaders)
    model=net(num_classes)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.fc.parameters(), lr=args.lr)
    
    train_with_early_stopping(model, dataset_loaders, args.epochs, loss_criterion, optimizer)
    save_model(model, args.model_dir)
    

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Training Job for Hyperparameter tuning")
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir-train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--data-dir-validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    args = parser.parse_args()
    
    main(args)