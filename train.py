import numpy as np
import pandas as pd
import os
import torch
import argparse
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Define constants
EPOCHS = 20
LR = 0.1
BATCH = 32
IM_SIZE = 224
STEP = 5
GAMMA = 0.2
DECAY = 0.9

class VehicleDataset(Dataset):
    def __init__(self, data, transform=None):
        super(VehicleDataset, self).__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        image, label = self.data.iloc[x, 0], self.data.iloc[x, -1]
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def create_df(base_dir, labeled=True):
    if labeled:
        dd = {"images": [], "labels": []}
        for i in os.listdir(base_dir):
            img_dirs = os.path.join(base_dir, i)
            for j in os.listdir(img_dirs):
                img = os.path.join(img_dirs, j)
                dd["images"] += [img]
                dd["labels"] += [i]
                
    else:
        dd = {"images": []}
        for i in os.listdir(base_dir):
            img_dirs = os.path.join(base_dir, i)
            dd["images"] += [img_dirs]
            
    return pd.DataFrame(dd)

def train_model(args):
    # Create output directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("Loading datasets...")
    train = create_df(os.path.join(args.data_dir, "train"))
    val = create_df(os.path.join(args.data_dir, "val"))
    
    # Process labels
    le = LabelEncoder()
    train["labels"] = le.fit_transform(train.loc[:, "labels"].values)
    val["labels"] = le.transform(val.loc[:, "labels"].values)

    # Save the label encoder
    with open(os.path.join(args.model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    
    NUM_CLASSES = train["labels"].nunique()
    print(f"Found {NUM_CLASSES} classes")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and data loaders
    train_ds = VehicleDataset(train, transform)
    val_ds = VehicleDataset(val, transform)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    print("Initializing model...")
    model = resnet50(weights='IMAGENET1K_V2' if not args.no_pretrained else None)
    num_ftrs = model.fc.in_features
    
    # Freeze base layers if using transfer learning
    if not args.no_pretrained:
        for param in model.parameters():
            param.requires_grad_ = False
    
    # Replace final layer
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    model.fc.requires_grad_ = True
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=GAMMA, step_size=STEP)
    
    # Initialize early stopping
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    
    # Training tracking variables
    best_model = deepcopy(model)
    best_acc = 0.0
    
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []
    
    # Training loop
    print("Starting training...")
    for i in range(1, args.epochs+1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        
        for data, label in train_dl:
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
                
            out = model(data)
            loss = criterion(out, label)
            train_loss += loss.item()
            train_acc += (out.argmax(1) == label).sum().item()
            train_total += out.size(0)
            loss.backward()
            optimizer.step()
            
        train_loss /= train_total
        train_acc /= train_total
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        
        with torch.no_grad():
            for data, label in val_dl:
                data, label = data.to(device), label.to(device)
                    
                out = model(data)
                loss = criterion(out, label)
                val_loss += loss.item()
                val_acc += (out.argmax(1) == label).sum().item()
                val_total += out.size(0)
                
        val_acc /= val_total
        val_loss /= val_total
        acc_train.append(train_acc)
        acc_val.append(val_acc)
        loss_train.append(train_loss)
        loss_val.append(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model)
            # Save best model so far
            torch.save(best_model.state_dict(), os.path.join(args.model_dir, "best_model_resnet50.pth"))
            print(f"Saved new best model with accuracy: {best_acc:.4f}")

        # Early stopping check
        if early_stopper.early_stop(val_loss):             
            print("Early stopping triggered")
            break
            
        print(f"Epoch {i} train loss {train_loss:.4f} acc {train_acc:.4f} val loss {val_loss:.4f} acc {val_acc:.4f}")
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model_resnet50.pth"))
    
    # Plot and save training curves
    epochs_range = list(range(1, len(acc_train) + 1))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    axes[0].plot(epochs_range, loss_train)
    axes[0].plot(epochs_range, loss_val)
    axes[0].set_title("Training and Validation loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend(["Training", "Validation"])
    
    axes[1].plot(epochs_range, acc_train)
    axes[1].plot(epochs_range, acc_val)
    axes[1].set_title("Training and Validation accuracies")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(["Training", "Validation"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Final model saved to {os.path.join(args.model_dir, 'final_model_resnet50.pth')}")
    print(f"Best model saved to {os.path.join(args.model_dir, 'best_model_resnet50.pth')}")
    print(f"Training curves saved to {os.path.join(args.output_dir, 'training_curves.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vehicle classification model")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Base directory for dataset")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=BATCH, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=10, help="Early stopping minimum delta")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no_pretrained", action="store_true", help="Don't use pretrained weights")
    
    args = parser.parse_args()
    train_model(args)