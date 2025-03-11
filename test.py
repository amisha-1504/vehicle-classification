import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as transforms

# Define constants
IM_SIZE = 224

def create_df(base_dir, labeled=False):
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
            img_path = os.path.join(base_dir, i)
            dd["images"] += [img_path]
            
    return pd.DataFrame(dd)

def predict_single(image_path, model, device, transform):
    """Predict a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        index = output.argmax(1).item()
        
    return index, probabilities.cpu().numpy()

def evaluate_model(args):
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Using device: {device}")
    
    # Load the test dataset
    print("Loading test dataset...")
    test = create_df(args.test_dir, labeled=args.labeled)
    
    # Load label encoder
    print("Loading label encoder...")
    with open(os.path.join(args.model_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load the model
    print("Loading model...")
    model = resnet50()
    num_ftrs = model.fc.in_features
    num_classes = len(le.classes_)
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_file), map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Make predictions
    print("Making predictions...")
    predicted_indices = []
    confidences = []
    
    for i, img_path in enumerate(test["images"]):
        idx, probs = predict_single(img_path, model, device, transform)
        predicted_indices.append(idx)
        confidences.append(probs)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(test)} images")
    
    # Convert indices to class names
    predicted_classes = le.inverse_transform(predicted_indices)
    
    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({
        "image_path": test["images"],
        "predicted_class": predicted_classes,
        "confidence": [np.max(conf) for conf in confidences]
    })
    
    predictions_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.csv')}")
    
    # If test dataset has labels, compute accuracy
    if args.labeled:
        test["labels"] = le.transform(test["labels"])
        correct = sum(predicted_indices == test["labels"])
        accuracy = correct / len(test)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test["labels"], predicted_indices)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        class_names = le.classes_
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
        
        # Create classification report
        report = classification_report(test["labels"], predicted_indices, target_names=class_names)
        with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        print(f"Classification report saved to {os.path.join(args.output_dir, 'classification_report.txt')}")
    
    # Visualize a grid of predictions
    rows, cols = args.grid_size, args.grid_size
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 16))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(test):
                img_path = test.iloc[idx, 0]
                img = plt.imread(img_path)
                
                # Add image
                axes[i][j].imshow(img)
                axes[i][j].set_title(f"Predicted: {predicted_classes[idx]}")
                axes[i][j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "prediction_grid.png"))
    print(f"Prediction grid saved to {os.path.join(args.output_dir, 'prediction_grid.png')}")
    
    # If in interactive mode, show the plot
    if args.interactive:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vehicle classification model")
    parser.add_argument("--test_dir", type=str, default="./dataset/test", help="Directory with test images")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory with saved model")
    parser.add_argument("--model_file", type=str, default="best_model_resnet50.pth", help="Model file name")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--labeled", action="store_true", help="Whether test data is labeled")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the visualization grid (NxN)")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    evaluate_model(args)