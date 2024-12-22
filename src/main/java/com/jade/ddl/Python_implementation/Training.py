from flask import Flask, request, jsonify
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore
from torchvision import datasets, transforms  # type: ignore
import sklearn.metrics as metrics
import traceback
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

app = Flask(__name__)

# Helper functions to dynamically create components

# Define the model
models = {}
aggregated_model = None


class DigitClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.LazyLinear(out_features=64)
        self.layer4 = nn.Linear(in_features=64, out_features=10)
        self.Relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.layer1(X)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.layer3(x)
        x = self.Relu(x)
        return self.layer4(x)


class DigitClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.LazyLinear(out_features=64)
        self.layer4 = nn.Linear(in_features=64, out_features=10)
        self.Relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.layer1(X)
        x = self.layer2(x)
        x = x.flatten(start_dim=1)
        x = self.layer3(x)
        x = self.Relu(x)
        return self.layer4(x)


def aggregate_models(models: list) -> nn.Module:
    """
    Aggregate the parameters of the models using FedAvg (Median Averaging).

    Args:
    - models: List of trained models (PyTorch nn.Module).

    Returns:
    - aggregated_model: The aggregated model with median parameters.
    """
    # Initialize the global model with the same architecture
    global_model = DigitClassifierModel()

    # Get the state_dicts (model parameters) from all models
    model_state_dicts = [model.state_dict() for model in models]

    # Initialize an empty dictionary to store the aggregated parameters
    aggregated_state_dict = {}

    # Iterate over each parameter in the model and average the corresponding parameters across models
    for key in model_state_dicts[0].keys():
        # Calculate the average of the parameter weights across all models
        param_list = [state_dict[key] for state_dict in model_state_dicts]
        aggregated_state_dict[key] = torch.median(
            torch.stack(param_list), dim=0)[0]

    # Load the aggregated weights into the global model
    global_model.load_state_dict(aggregated_state_dict)

    return global_model


def test_model(model, X_test, y_test):
    """
    Test the model and calculate accuracy, precision, recall, and F1 score.

    Args:
    - model: Trained PyTorch model to evaluate.
    - X_test: Test dataset (features) as a PyTorch Tensor.
    - y_test: Test labels as a PyTorch Tensor.

    Returns:
    - metrics_dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        outputs = model(X_test)  # Forward pass
        # Apply Softmax to get probabilities
        preds = nn.Softmax(dim=1)(outputs)
        preds = torch.argmax(preds, dim=1)  # Get class predictions (argmax)
        all_preds = preds.cpu().numpy()  # Convert predictions to numpy

    y_test = y_test.cpu().numpy()  # Convert ground truth to numpy

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, all_preds)
    precision = metrics.precision_score(y_test, all_preds, average="weighted")
    recall = metrics.recall_score(y_test, all_preds, average="weighted")
    f1 = metrics.f1_score(y_test, all_preds, average="weighted")

    # Return metrics as a dictionary
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics_dict


def build_model(n):
    if len(list(models.keys())) == 3:
        model = models[n]
    else:
        model = DigitClassifierModel()
    return model


def build_optimizer(model, config):
    """
    Build optimizer dynamically based on configuration.
    """
    optimizer_type = config['type']
    lr = config['learning_rate']
    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_loss_function(config):
    """
    Build loss function dynamically based on configuration.
    """
    if config == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif config == 'MSELoss':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {config}")


def prepare_dataloader(data_config, b_size=32):
    """
    Create a DataLoader from given data.
    """
    inputs = data_config['X_train']
    targets = data_config['y_train']
    # print('batch size :{}'.format(b_size))

    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=32, shuffle=True)


data_dic = {}
X_test = None
y_test = None


def Data_split(dataset, num_worker):
    if dataset == "mnist":
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])

        # Download and load the MNIST dataset
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        X_train = train_dataset.data.numpy()  # (60000, 28, 28)
        y_train = train_dataset.targets.numpy()  # (60000,)

        # Reshape to add the channel dimension (1 for grayscale)
        X_train = torch.from_numpy(X_train).unsqueeze(
            dim=1).float()  # Shape: (60000, 1, 28, 28)
        y_train = torch.from_numpy(y_train).long()  # Shape: (60000,)

        global X_test
        global y_test
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        shape_for_one = int(X_train.shape[0]/num_worker)

        X_train1 = X_train[:shape_for_one]
        X_train2 = X_train[shape_for_one:shape_for_one*2]
        X_train3 = X_train[shape_for_one*2:]

        y_train1 = y_train[:shape_for_one]
        y_train2 = y_train[shape_for_one:shape_for_one*2]
        y_train3 = y_train[shape_for_one*2:]

        data_dic[1] = {'X_train': X_train1, 'y_train': y_train1}
        data_dic[2] = {'X_train': X_train2, 'y_train': y_train2}
        data_dic[3] = {'X_train': X_train3, 'y_train': y_train3}
        print("x1 shape ", X_train1.shape)
        print("x2 shape ", X_train2.shape)
        print("x3 shape ", X_train3.shape)
        print("y1 shape ", y_train1.shape)
        print("y2 shape ", y_train2.shape)
        print("y3 shape ", y_train3.shape)


@app.route('/split', methods=['POST'])
def split():
    try:
        # Parse incoming JSON configuration
        data = request.json
        print("Training request received.")

        # Extract configurations
        dataset = data['dataset']
        num_worker = data['num_worker']

        print('the Data set name : '+dataset)
        print('the Number of trainers : '+str(num_worker))

        Data_split(dataset, num_worker)

        return jsonify({'status': 'success'})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/train/<int:trainer_id>', methods=['POST'])
def train(trainer_id):

    try:
        # Parse incoming JSON configuration
        data = request.json
        print("Training request received.")

        # Extract configurations
        optimizer_config = data['optimizer']
        loss_config = data['loss_function']
        training_config = data['training']

        # Build components
        model = build_model(trainer_id)
        optimizer = build_optimizer(model, optimizer_config)
        loss_function = build_loss_function(loss_config)
        print('trainer id : {} <----------\n'.format(trainer_id))
        dataloader = prepare_dataloader(
            data_dic[trainer_id], b_size=training_config["batch_size"])
        print('Data Loader : ', dataloader)

        # # Training loop
        num_epochs = int(training_config['epochs'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train()
        # history = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        models[trainer_id] = model
        return jsonify({'status': 'success'})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/aggregate', methods=['GET'])
def aggregate():
    global aggregated_model
    models_list = []
    for i in list(models.keys()):
        models_list.append(models[i])
    aggregated_model = aggregate_models(models_list)
    return jsonify({'status': 'success'})


@app.route('/test', methods=['GET'])
def test():
    global aggregated_model
    global X_test
    global y_test

    if aggregated_model != None:
        if X_test != None and y_test != None:
            result = test_model(aggregated_model, X_test=X_test, y_test=y_test)
            print(f'Result :: {result}')
            return jsonify({'status': 'success', 'result': result})
        else:
            print('No Test data found !!!')
            return jsonify({'status': "Failed"}, 404)
    else:
        print("No Aggregated model found !!!")
        return jsonify({'status': "Failed"}, 404)


if __name__ == '__main__':
    app.run(port=5000)
