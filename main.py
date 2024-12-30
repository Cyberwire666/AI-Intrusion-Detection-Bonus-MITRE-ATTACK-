# Import required libraries
import pandas as pd  # Pandas for data manipulation (handling datasets)
import numpy as np  # Numpy for numerical operations (working with arrays and matrices)
import torch  # PyTorch for building and training the neural network (core ML framework)
import torch.nn as nn  # PyTorch neural network module (for defining and using models)
import torch.optim as optim  # Optimizers from PyTorch (for training models)
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling numerical features and encoding categorical variables
from sklearn.model_selection import train_test_split  # For splitting the data into training and test sets
from sklearn.metrics import accuracy_score  # To evaluate performance of the model
import matplotlib.pyplot as plt  # For plotting graphs (used for loss and accuracy)
import seaborn as sns  # For better visualization (particularly for bar plots)

# Load the dataset containing intrusion detection data
df = pd.read_csv('intrusion_detection_data.csv')  # Load the CSV file into a Pandas DataFrame
# Print the first few rows of the dataset to understand its structure
print("-------------------------------------START----------------------------------------------")
print(f"Sample data:\n{df.head()}")  # Display the first five rows to preview the data

# Shuffle the data to avoid any bias caused by ordered rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data randomly


# Separate the features (X) and target variable (y)
X = df.drop(['attack'], axis=1)  # Drop attack and mitre_attack columns from features (X)
y = df['attack']  # Use the 'attack' column as the target variable (y)

# Split the dataset into train and test sets (30% for testing, 70% for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Split the data

# Preprocessing: Drop unnecessary columns that cannot be used by the model (like source_ip and destination_ip)
X_train = X_train.drop(columns=['source_ip', 'destination_ip'], errors='ignore')  # Drop irrelevant columns
X_test = X_test.drop(columns=['source_ip', 'destination_ip'], errors='ignore')  # Drop irrelevant columns

# Apply Label Encoding for categorical features (service, protocol_type, flag) to convert them into numeric format
label_encoders = {}  # Dictionary to store label encoders for each column
categorical_columns = ['service', 'protocol_type', 'flag']  # List of categorical columns to encode
for col in categorical_columns:
    le = LabelEncoder()  # Initialize the label encoder
    X_train[col] = le.fit_transform(X_train[col])  # Fit and transform the categorical feature on the training data
    X_test[col] = le.transform(X_test[col])  # Transform the test data based on training data encoding
    label_encoders[col] = le  # Store the encoder for future use

# Standardize numerical features (scaling them to have mean=0 and variance=1)
scaler = StandardScaler()  # Initialize the StandardScaler to scale numerical features
X_train = scaler.fit_transform(X_train)  # Fit and transform the training set (scaling it)
X_test = scaler.transform(X_test)  # Only transform the test set, using parameters from training data

# Convert data to PyTorch tensors for model input
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Convert the training features to tensor
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Convert target variable to tensor (unsqueezing for correct shape)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # Convert the test features to tensor
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)  # Convert test target to tensor

# Neural Network Model Definition
class IntrusionDetectionNN(nn.Module):  # Define a custom neural network class inheriting from nn.Module
    def __init__(self, input_size):  # Constructor to initialize layers
        super(IntrusionDetectionNN, self).__init__()
        # Define a 3-layer fully connected neural network architecture
        self.fc1 = nn.Linear(input_size, 64)  # Input layer to hidden layer with 64 neurons
        self.fc2 = nn.Linear(64, 32)  # First hidden layer to second hidden layer with 32 neurons
        self.fc3 = nn.Linear(32, 1)  # Second hidden layer to output layer (binary output)
        self.sigmoid = nn.Sigmoid()  # Sigmoid function for binary classification

    def forward(self, x):
        # Define the forward pass using ReLU activations for hidden layers and Sigmoid for output
        x = torch.relu(self.fc1(x))  # ReLU activation for first hidden layer
        x = torch.relu(self.fc2(x))  # ReLU activation for second hidden layer
        return self.sigmoid(self.fc3(x))  # Apply Sigmoid activation for final output

# Initialize model with input size equal to the number of features
model = IntrusionDetectionNN(X_train.shape[1])  # Initialize model based on input size (number of features in X_train)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer for weight updates, learning rate set to 0.001

# Training Loop
epochs = 20  # Number of epochs for training (iterations over the entire dataset)
batch_size = 32  # Batch size for training (number of samples processed in each batch)
train_losses, test_losses = [], []  # List to store loss values during training
train_accuracies, test_accuracies = [], []  # List to store accuracy values

# Training the model
for epoch in range(epochs):  # Loop through the specified number of epochs
    model.train()  # Set model to training mode (this enables dropout, batch normalization, etc.)
    permutation = torch.randperm(X_train_tensor.size(0))  # Randomly shuffle the dataset for batch processing
    epoch_train_loss = 0  # Initialize total training loss for this epoch

    # Train in batches
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]  # Get indices for the current batch
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]  # Extract the features and target for the batch

        optimizer.zero_grad()  # Reset gradients to zero for each batch
        outputs = model(batch_X)  # Perform a forward pass and get predictions
        loss = criterion(outputs, batch_y)  # Calculate loss (error between predictions and true values)
        loss.backward()  # Backpropagate the loss (compute gradients)
        optimizer.step()  # Perform an optimization step (adjust weights)

        epoch_train_loss += loss.item()  # Add the batch loss to the total loss for the epoch

    # Store average training loss for the epoch
    train_losses.append(epoch_train_loss / len(X_train_tensor))

    # Calculate train accuracy
    with torch.no_grad():  # Disable gradient tracking for evaluation (faster inference)
        train_predictions = (model(X_train_tensor) > 0.5).float()  # Make binary predictions (1 if probability > 0.5, else 0)
        train_accuracy = accuracy_score(y_train_tensor.numpy(), train_predictions.numpy())  # Calculate accuracy
        train_accuracies.append(train_accuracy)

    # Evaluate on the test set
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = model(X_test_tensor)  # Get predictions for the test set
        test_loss = criterion(test_outputs, y_test_tensor)  # Calculate test loss
        test_losses.append(test_loss.item())  # Append test loss to the list
        test_predictions = (test_outputs > 0.5).float()  # Make binary predictions for test data
        test_accuracy = accuracy_score(y_test_tensor.numpy(), test_predictions.numpy())  # Calculate test accuracy
        test_accuracies.append(test_accuracy)

    # Print epoch statistics
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
    print(f"Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

# Plot Loss and Accuracy over epochs
plt.figure(figsize=(12, 6))  # Set up a figure for the plots

# Loss plot
plt.subplot(1, 2, 1)  # Create the first subplot for loss
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')  # Plot train loss
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='o')  # Plot test loss
plt.xlabel('Epochs')  # Label for the x-axis
plt.ylabel('Loss')  # Label for the y-axis
plt.title('Train and Test Loss')  # Title for the loss plot
plt.legend()  # Display the legend

# Accuracy plot
plt.subplot(1, 2, 2)  # Create the second subplot for accuracy
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', marker='o')  # Plot train accuracy
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', marker='o')  # Plot test accuracy
plt.xlabel('Epochs')  # Label for the x-axis
plt.ylabel('Accuracy')  # Label for the y-axis
plt.title('Train and Test Accuracy')  # Title for the accuracy plot
plt.legend()  # Display the legend

plt.tight_layout()  # Adjust the layout to prevent overlap
plt.show()  # Show the plots
print("--------------------------------------------END--------------------------------------------------")



#----------------------------------------------BONUS---------------------------------------------------------------
""" # Visualize the distribution of MITRE ATT&CK techniques
mitre_counts = df['mitre_attack'].value_counts()  # Count occurrences of each MITRE ATT&CK technique
plt.figure(figsize=(12, 8))  # Set the figure size for the bar plot
sns.barplot(x=mitre_counts.values, y=mitre_counts.index)  # Plot a bar chart of technique frequencies
plt.title("MITRE ATT&CK Technique Distribution")  # Set the plot title
plt.xlabel("Frequency")  # Label for the x-axis
plt.ylabel("MITRE ATT&CK Techniques")  # Label for the y-axis
plt.show()  # Show the plot
 """

# Function to map real-time data to MITRE ATT&CK techniques
# Define a function to map the features of the dataset to MITRE ATT&CK techniques
""" def map_to_mitre(row):
    if row['attack'] == 1:  # Check if the attack label is 1 (positive class for attack)
        # Check different conditions for MITRE ATT&CK tactic and technique mappings
        if row['protocol_type'] == 'udp' and row['service'] == 'ftp':
            return "Tactic: Exfiltration, Technique: Exfiltration Over Alternative Protocol"
        elif row['flag'] == 'FIN':
            return "Tactic: Command and Control, Technique: Application Layer Protocol"
        elif row['packet_loss'] > 0.01:
            return "Tactic: Impact, Technique: Data Destruction"
        elif row['packet_count'] > 1000 and row['total_bytes'] > 100000:
            return "Tactic: Initial Access, Technique: Exploit Public-Facing Application"
        else:
            return "Tactic: Discovery, Technique: Network Service Scanning"
    else:
        return "Tactic: None, Technique: None"  # If no attack, return no technique mapping

# Apply the function to each row of the dataset to create the 'mitre_attack' column
df['mitre_attack'] = df.apply(map_to_mitre, axis=1)  # Map MITRE ATT&CK technique labels for all rows

# Example input for real-time prediction
live_input = [
    [300, 'udp', 'ftp', 'SF', 10000, 5000, 0, 0.01, 0, 2, 0, 0, 0, 0]
]  # Example of a single data sample with same structure as dataset

# Get MITRE ATT&CK tactic and technique prediction for the new input
mapped_tactic = map_to_mitre(live_input)
print(f"Predicted MITRE ATT&CK Technique: {mapped_tactic}")
 """