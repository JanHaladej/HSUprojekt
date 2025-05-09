import os
import torch
import matplotlib.pyplot as plt
import json
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import random_split, DataLoader

class ModelTrainer:
    def __init__(self, dataset, model_class, model_params, training_params, output_dir="TrainingResults"):
        self.dataset = dataset
        self.model_class = model_class
        self.model_params = model_params
        self.training_params = training_params
        
        param_summary = f"conv{model_params.get('num_conv_layers', '')}_do{int(model_params.get('use_dropout', False))}_lr{training_params.get('learning_rate', 0):.0e}_bs{training_params.get('batch_size', 0)}"
        self.output_dir = os.path.join(output_dir, param_summary)
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        train_size = int(self.training_params["train_split"] * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        
        num_workers = os.cpu_count() or 4

        self.train_loader = DataLoader(train_dataset, batch_size=self.training_params["batch_size"], shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.training_params["batch_size"], shuffle=False, num_workers=num_workers)

    def _init_model(self):
        self.model = self.model_class(**self.model_params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_params["learning_rate"])

    def train(self):
        best_loss = float("inf")
        epochs_no_improve = 0

        train_losses = []
        val_losses = []
        
        for epoch in range(self.training_params["epochs"]):
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            train_losses.append(train_loss)

            val_loss = self._evaluate_loss()
            val_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{self.training_params['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.training_params["patience"]:
                print("Early stopping triggered.")
                break

        self._plot_losses(train_losses, val_losses)
        self._final_evaluation()

    def _evaluate_loss(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def _plot_losses(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label='Train Loss', color='orange')
        plt.plot(val_losses, label='Validation Loss', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()

    def _final_evaluation(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_model.pth")))
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        with open(os.path.join(self.output_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

        with open(os.path.join(self.output_dir, "params.json"), "w") as f:
            json.dump({
                "model_params": self.model_params,
                "training_params": self.training_params
            }, f, indent=4)

        print("Final Evaluation:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
