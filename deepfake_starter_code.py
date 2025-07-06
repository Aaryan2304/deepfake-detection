# deepfake_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class DeepfakeDetector(nn.Module):
    def __init__(self, backbone='efficientnet', num_classes=2, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        
        # Initialize num_features with a default value
        num_features = 1280  # Default value for efficientnet-b0
        
        if backbone == 'efficientnet':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'efficientnet' or 'resnet50'")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = self.labels[idx]
        return image, label

class DeepfakeTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        return total_loss / len(self.val_loader), accuracy
    
    def train(self, epochs):
        best_accuracy = 0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_deepfake_model.pth')
            
            self.scheduler.step()
        
        print(f'Best validation accuracy: {best_accuracy:.4f}')

class DeepfakeEvaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        auc_roc = roc_auc_score(all_targets, all_probs)
        
        print(f'Test Results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'AUC-ROC: {auc_roc:.4f}')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc_roc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }

class DeepfakeExplainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Get the last convolutional layer for Grad-CAM
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'features'):
                self.target_layer = [model.backbone.features[-1]]
            else:
                self.target_layer = [model.backbone.layer4[-1]]
        
        self.cam = GradCAM(model=model, target_layers=self.target_layer)
    
    def explain_prediction(self, image, target_class=None):
        """Generate Grad-CAM visualization for a single image"""
        # Prepare input
        input_tensor = image.unsqueeze(0).to(self.device)
        
        # Get model prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Convert to numpy for visualization
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam
    
    def visualize_explanation(self, original_image, grayscale_cam, save_path=None):
        """Visualize the Grad-CAM overlay on original image"""
        # Normalize original image to [0, 1] range
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Create visualization
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(grayscale_cam, cmap='hot')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(visualization)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.tight_layout()
        plt.show()

# Data preprocessing transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.1),
        A.HueSaturationValue(p=0.1),
        A.RandomGamma(p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DeepfakeDetector(backbone='efficientnet', num_classes=2)
    
    # Load data (you'll need to implement data loading based on your dataset)
    train_transform, val_transform = get_transforms()
    
    # Create datasets and loaders
    # train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform)
    # val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    # trainer = DeepfakeTrainer(model, train_loader, val_loader)
    
    # Train model
    # trainer.train(epochs=50)
    
    # Evaluate model
    # evaluator = DeepfakeEvaluator(model, test_loader)
    # results = evaluator.evaluate()
    
    # Explain predictions
    # explainer = DeepfakeExplainer(model)
    # cam = explainer.explain_prediction(test_image)
    # explainer.visualize_explanation(original_image, cam)
    
    print("Deepfake detection system initialized successfully!")
