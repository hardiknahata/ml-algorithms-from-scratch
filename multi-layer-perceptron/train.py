import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from model import SimpleNN


class ModelTrainer:
    def __init__(self, batch_size=64, num_epochs=100, learning_rate=0.001, device='cpu'):
        # Configuration
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Data Preparation
        self.train_loader, self.val_loader, self.input_size, self.output_size = self._prepare_data()

        # Model, Loss, Optimizer
        self.model = SimpleNN(self.input_size, 64, self.output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _prepare_data(self):
        print("Loading and preparing the dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)

        input_size = X_train.shape[1]
        output_size = len(iris.target_names)

        return train_loader, val_loader, input_size, output_size

    def train(self):
        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] - Average Loss: {avg_loss:.4f}")
                self._validate()

        duration = time.time() - start_time
        print("Training complete.")
        print(f"Duration: {duration:.2f} seconds, Device: {self.device}")

    def _validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%\n")


if __name__ == "__main__":
    trainer = ModelTrainer(batch_size=64, num_epochs=100, learning_rate=0.001)
    trainer.train()