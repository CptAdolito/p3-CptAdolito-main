# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm

# own modules
from src.models import CNNModel
from src.data import load_data
from src.utils import (
    Accuracy,
    save_model,
    set_seed,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set all seeds and number of threads
set_seed(42)
torch.set_num_threads(8)

# Static variables
DATA_PATH = "data"
NUMBER_OF_CLASSES = 10


def main():
    """
    Main training function
    """
    # Load data
    train_loader, val_loader, _ = load_data(DATA_PATH, batch_size=64)

    # Initialize model, loss function, and optimizer
    model = CNNModel(num_classes=NUMBER_OF_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/imagenette_experiment")

    # Training loop
    num_epochs = 50
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        running_loss = 0.0
        train_acc = Accuracy()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_acc.update(outputs, labels)

        avg_loss = running_loss / len(train_loader)
        train_accuracy = train_acc.compute()

        # Validation loop
        model.eval()
        val_acc = Accuracy()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_acc.update(outputs, labels)

        val_accuracy = val_acc.compute()

        # Logging
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, \
             Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, "best_model")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
