"""
TO BE DONE LATER!!!
try this data augmentation and then check the model accuracy again after adding more images in the dataset by doing DA.
Compare this with the model accuracy trained without augmented data, and check if the accuracy of the model increases.

"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from model_history.model_1 import ObjectClassification
import torch
from sklearn.metrics import confusion_matrix
import time

from collections import Counter

since = time.time()

training_transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    "../../datasets/objects/train",
    transform=training_transform
)

val_dataset = datasets.ImageFolder(
    "../../datasets/objects/validation",
    transform=val_transform
)

train_dataset_counts = Counter(train_dataset.targets)
validation_dataset_counts = Counter(val_dataset.targets)

total_training_images = len(train_dataset)
training_weights = []
total_validation_images = len(val_dataset)
validation_weights = []

num_classes = len(train_dataset.classes)

for i in range(num_classes):
    training_weights.append(total_training_images / (num_classes * train_dataset_counts[i]))
    validation_weights.append(total_validation_images / (num_classes * validation_dataset_counts[i]))

print("training dataset" ,train_dataset.class_to_idx)
print("dataset count" ,train_dataset_counts)
print("weights" ,training_weights)
print("validation dataset" ,train_dataset.class_to_idx)
print("dataset count" ,validation_dataset_counts)
print("weights" ,validation_weights)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

model = ObjectClassification()

training_weights = torch.tensor(training_weights)
validation_weights = torch.tensor(validation_weights)

criterion_training = nn.CrossEntropyLoss(weight=training_weights)
criterion_validation = nn.CrossEntropyLoss(weight=validation_weights)


optimizer = Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

best_acc = 0

for epoch in range (300):
    
    #------training-------

    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion_training(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Compute training accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []


    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            loss = criterion_validation(logits, labels)
            val_loss +=  loss.item()
            # probabilities = torch.softmax(logits, dim=1)
            # predictions = torch.argmax(probabilities, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predictions == labels).sum().item()

            total += labels.size(0)
    
    print(confusion_matrix(all_labels, all_preds))

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new model! ")

print(f"time taken: {time.time() - since:.2f} secs")

torch.save(model.state_dict(), "model.pth")


