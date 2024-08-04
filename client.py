import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import requests
from model import SimpleCNN
import io

def train_on_device(model, device_data, epochs=1, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(device_data, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# Client configuration
SERVER_URL = "http://server_ip:5000"  # Replace with your server's IP

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_data = datasets.MNIST('data', train=True, download=True, transform=transform)
device_data, _ = random_split(mnist_data, [30000, 30000])

# Load test data
test_data = datasets.MNIST('data', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

for round in range(5):
    print(f"Round {round + 1}")
    
    # Get global model
    print(f"Attempting to connect to {SERVER_URL}")
    response = requests.get(f"{SERVER_URL}/get_model")
    model = SimpleCNN()
    model.load_state_dict(torch.load(io.BytesIO(response.content)))
    print("Global model received")
    
    # Train locally
    print("Starting local training")
    local_state = train_on_device(model, device_data)
    print("Local training completed")
    
    # Send updated model to server
    print("Sending updated model to server")
    buffer = io.BytesIO()
    torch.save(local_state, buffer)
    requests.post(f"{SERVER_URL}/update_model", data=buffer.getvalue())
    print("Model sent to server")
    
    # Evaluate model
    model.load_state_dict(local_state)
    accuracy = evaluate_model(model, test_loader)
    print(f"Local model accuracy: {accuracy:.4f}")

print("Federated learning completed")