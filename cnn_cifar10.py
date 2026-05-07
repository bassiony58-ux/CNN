# ============================================================
#  CNN Image Classifier — CIFAR-10 | PyTorch
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# 0. Reproducibility & Device
# ─────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Hyper-parameters
# ─────────────────────────────────────────────
NUM_CLASSES  = 10
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
EPOCHS       = 10

# ─────────────────────────────────────────────
# 2. Dataset — CIFAR-10  (3 × 32 × 32)
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),   # normalise each channel
                         std =(0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True,  download=True, transform=transform
)
test_dataset  = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASSES = train_dataset.classes   # 10 class names
print(f"Classes: {CLASSES}\n")

# ─────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────
class CNN(nn.Module):
    """
    Three convolutional blocks followed by two fully-connected layers.

    Input  : (B, 3, 32, 32)
    Output : (B, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()

        # ── Conv Block 1 ── 3 → 16 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32×32 → 16×16
        )

        # ── Conv Block 2 ── 16 → 32 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16×16 → 8×8
        )

        # ── Conv Block 3 ── 32 → 64 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 8×8 → 4×4
        )

        # ── Flatten + Fully-Connected Layers ──
        self.flatten = nn.Flatten()                  # 64 × 4 × 4 = 1024

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),             # raw logits (no softmax)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


# Instantiate model and move to device
model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
print("Model architecture:")
print(model)

# ─────────────────────────────────────────────
# 4. Loss Function & Optimizer
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ─────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────
def train_one_epoch(epoch: int) -> tuple[float, float]:
    """Train for one epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # clear gradients
        outputs = model(images)         # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # backprop
        optimizer.step()               # update weights

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


print("\n" + "="*55)
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc (%)':>14}")
print("="*55)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    print(f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>14.2f}")

# ─────────────────────────────────────────────
# 6. Evaluation on Test Set
# ─────────────────────────────────────────────
def evaluate() -> float:
    """Return test accuracy (%)."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total


test_accuracy = evaluate()

print("="*55)
print(f"\n✅  Final Test Accuracy: {test_accuracy:.2f}%")
print("="*55)
