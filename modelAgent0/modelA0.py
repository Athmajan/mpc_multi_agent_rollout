import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import wandb
from tqdm import tqdm
# from .getExpReplay import fetch_filtered_documents
import os
from torch.optim.lr_scheduler import StepLR

# Custom Dataset
class ObservationDataset(Dataset):
    """Dataset wrapper for observation-action pairs.

    This dataset expects an iterable of dict-like entries where each entry
    contains 'prev_observations' (features) and 'actions' (targets).

    Attributes:
        data (list): Original list of data entries.
        X (torch.Tensor): Tensor of input observations (float32).
        y (torch.Tensor): Tensor of target actions (float32).
    """
    def __init__(self, data):
        self.data = data
        self.X = torch.tensor([d['prev_observations'] for d in data], dtype=torch.float32)
        self.y = torch.tensor([d['actions'] for d in data], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Neural Network
class ActionPredictor_BN(nn.Module):
    """Feed-forward action predictor network with Batch Normalization.

    This model maps flattened observation inputs to predicted actions using
    two hidden layers with BatchNorm, ReLU activations, and a final sigmoid.

    Args:
        input_size (int): Dimensionality of the input observation vector.
        output_size (int): Dimensionality of the output action vector.
    """
     

    def __init__(self, input_size, output_size):
        super(ActionPredictor_BN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
# Neural Network
class ActionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
# Neural Network
class ActionPredicton_DROP(nn.Module):
    """Smaller feed-forward action predictor without Batch Normalization.

    This model uses two hidden layers of size 32 and a final sigmoid activation.

    Args:
        input_size (int): Dimensionality of the input observation vector.
        output_size (int): Dimensionality of the output action vector.
    """


    def __init__(self, input_size, output_size,dropoutRate):
        super(ActionPredicton_DROP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropoutRate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x) 
        x = self.fc3(x)
        return x

def trainModel(batch_size,lr,epochs,wdecay,batchNorm,dropoutRate=None):
    """Train an action prediction model on observation-action pairs.

    The function:
    - Loads data via fetch_filtered_documents (expected to return a list of dicts
      with 'prev_observations' and 'actions').
    - Wraps data into ObservationDataset and splits into train/test (90/10).
    - Constructs a model according to batchNorm and dropoutRate flags.
    - Trains using MSELoss and Adam optimizer while logging losses to Weights & Biases.
    - Saves model checkpoints periodically.

    Args:
        batch_size (int): Batch size for training and evaluation DataLoaders.
        lr (float): Initial learning rate for the optimizer.
        epochs (int): Number of training epochs.
        wdecay (float): Weight decay (L2 regularization) for the optimizer.
        batchNorm (bool): If True, use ActionPredictor_BN; otherwise use ActionPredictor.
        dropoutRate (float or None): If set to a float in (0,1), use ActionPredicton_DROP with this dropout probability.

    Returns:
        None

    Notes:
        - This function expects fetch_filtered_documents to be available in the import path.
        - Model checkpoints are saved to a hard-coded path; adjust pathDir as required.
        - wandb.run is expected to be initialized before calling this function (the script's main block does that).
    """


    data = fetch_filtered_documents("seq_roll_2_4_10_20_3")
    dataset = ObservationDataset(data)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_size = len(data[0]['prev_observations'])
    output_size = len(data[0]['actions'])

    if batchNorm:
        model = ActionPredictor_BN(input_size, output_size)
    else:
        model = ActionPredictor(input_size, output_size)
    
    if dropoutRate is not None:
        model = ActionPredicton_DROP(input_size, output_size,dropoutRate)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scheduler = StepLR(optimizer, step_size=20000, gamma=0.1)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)

        test_loss /= len(test_loader.dataset)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})

        scheduler.step()


        # Save the model every 100 epochs
        if (epoch + 1) % 5000 == 0 or epoch == 0:
            # Update the model save path to include the run name
            pathDir = 'path'
            model_path = pathDir + f"action_predictor_{wandb.run.name}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)

    
    wandb.finish()
    
if __name__ == "__main__":
    from getExpReplay import fetch_filtered_documents
    batch_size = 500
    lr = 0.0001
    epochs = 100000
    wdecay = 0.0001
    batchNorm = True
    dropoutRate = None

    wandb.init(project="action-prediction", 
               config={"batch_size": batch_size, 
                       "lr": lr, 
                       "epochs": epochs,
                       "wdecay":wdecay,
                       "batchNormFlag" : batchNorm,
                       "dropout" : dropoutRate,
                       })
    wandb.config.batch_size = batch_size
    wandb.config.lr = lr
    wandb.config.epochs = epochs
    wandb.config.wdecay = wdecay
    wandb.config.batchNormFlag = batchNorm
    wandb.config.dropout = dropoutRate

    wandb_artifact = wandb.Artifact(
        name=f"trainModel",
        type="source_code")
    py_files = [
        "file",  # Replace with the main script filename
    ]
    for file in py_files:
        if os.path.exists(file):
            wandb_artifact.add_file(file)
        else:
            print(f"Warning: {file} not found. Skipping.")
    wandb.log_artifact(wandb_artifact)
    


    trainModel(batch_size,lr,epochs,wdecay,batchNorm,dropoutRate)

    


        

    
