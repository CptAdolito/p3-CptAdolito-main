# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# own modules
from src.data import load_data
from src.utils import (
    Accuracy,
    load_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO
    # Load test data
    _, _, test_loader = load_data(DATA_PATH, batch_size=64)

    # Load trained model
    model: RecursiveScriptModule = load_model(f"{name}").to(device)
    model.eval()

    # Accuracy metric
    test_acc = Accuracy()

    # Evaluation loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_acc.update(outputs, labels)

    accuracy = test_acc.compute()
    return accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
