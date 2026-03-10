import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
from utils.model import ResNet9
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Data Augmentation
# ----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

train_data = datasets.ImageFolder(root='Data/train',
                                  transform=train_transform)
val_data = datasets.ImageFolder(root='Data/val',
                                transform=val_transform)

train_loader = DataLoader(train_data,
                          batch_size=16,
                          shuffle=True,
                          pin_memory=False)

print("Train loader created")

val_loader = DataLoader(val_data,
                        batch_size=32,
                        shuffle=False)

# ----------------------------
# Device Setup
# ----------------------------
device = torch.device("cuda"
          if torch.cuda.is_available()
          else "cpu")

num_classes = len(train_data.classes)
model = ResNet9(3, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=1e-3,
                       weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=7,
            gamma=0.1)

# ----------------------------
# Training
# ----------------------------
num_epochs = 3
best_acc = 0

with open('training_log.csv',
          mode='w',
          newline='') as file:

    writer = csv.writer(file)
    writer.writerow(["epoch",
                     "train_loss",
                     "val_loss",
                     "train_acc",
                     "val_acc"])
    print("Starting training loop...")
    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images,labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _,preds = torch.max(outputs,1)
            total += labels.size(0)
            correct += (preds==labels).sum().item()

        train_loss = running_loss/len(train_loader)
        train_acc = correct/total

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images,labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs,labels)

                val_loss += loss.item()

                _,preds = torch.max(outputs,1)

                val_total += labels.size(0)
                val_correct += (preds==labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss/len(val_loader)
        val_acc = val_correct/val_total

        scheduler.step()

        writer.writerow([epoch+1,
                         train_loss,
                         val_loss,
                         train_acc,
                         val_acc])

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")

        # ---- SAVE BEST MODEL ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       'best_model.pth')

# ----------------------------
# Evaluation
# ----------------------------
print("\nClassification Report:\n")
print(classification_report(all_labels,
                            all_preds,
                            target_names=train_data.classes))

cm = confusion_matrix(all_labels,
                      all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("\n✅ Training Completed!")
print("Best Model Saved as best_model.pth")