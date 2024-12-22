# Distributed Deep Learning with JADE Agents

## Overview
This project implements a **distributed deep learning framework** using JADE (Java Agent DEvelopment Framework). The system leverages multiple agents to collaboratively train a neural network model on the MNIST dataset. The framework simulates a federated learning-like environment where agents are responsible for specific tasks such as data distribution, model training, aggregation, and testing.

## Architecture
The system consists of the following agents:

### 1. **Coordinator Agent**
- Oversees the entire training process.
- Initializes and monitors the other agents.
- Controls the workflow, ensuring smooth communication between agents.

### 2. **Data Distributor Agent**
- Prepares and distributes the MNIST dataset.
- Splits the dataset into subsets for training agents.
- Sends data to the training agents.

### 3. **Trainer Agents (3)**
- Receive data subsets from the Data Distributor Agent.
- Train individual models on their respective subsets.
- Send trained models back to the Model Aggregator Agent.

### 4. **Model Aggregator Agent**
- Receives models from Trainer Agents.
- Aggregates the models into a single global model using federated averaging.
- Sends the aggregated model to the Test Agent.

### 5. **Test Agent**
- Evaluates the aggregated model on the test dataset.
- If the accuracy is greater than 80%, the training is deemed successful.
- If accuracy is below 80%, notifies the Coordinator Agent to initiate another round of training.

## Workflow
1. **Initialization**: The Coordinator Agent launches all other agents and prepares the system.
2. **Data Distribution**: The Data Distributor Agent splits the MNIST dataset and distributes it to the Trainer Agents.
3. **Model Training**: The Trainer Agents independently train models on their respective datasets.
4. **Model Aggregation**: The Model Aggregator Agent collects the trained models and performs aggregation.
5. **Model Testing**: The Test Agent evaluates the aggregated model.
    - If the accuracy exceeds 80%, the process concludes.
    - Otherwise, a new training iteration begins.

## Dataset
- **MNIST Dataset**: A collection of handwritten digits (0-9) commonly used for machine learning and deep learning experiments.
- The dataset is split into training and testing subsets during the process.

## Technologies Used
- **JADE**: For agent-based communication and task coordination.
- **Deep Learning Framework**: For model creation and training.
- **Java**: Primary programming language for agent implementation.

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/MohamedGaamouci/Distributed-Deep-Learning-With-Agent-jade.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Distributed-Deep-Learning-With-Agent-jade
    ```
3. Compile the JADE project.
4. Run the system:
    - Start the Coordinator Agent.
    - The other agents will automatically initialize and perform their tasks.

## Results
- The system aims for a model accuracy greater than 80%.
- If the accuracy threshold is not met, additional training iterations will be performed until the desired accuracy is achieved.

## Future Enhancements
- Extend support for larger and more complex datasets.
- Implement secure data transfer between agents.
- Introduce fault-tolerance mechanisms for agent failures.

## Contact
For any inquiries or contributions, please contact [Mohamed Gaamouci](https://github.com/MohamedGaamouci).

