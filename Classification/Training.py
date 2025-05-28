import os
from typing import Counter
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.utils import compute_class_weight
from collections import defaultdict
import random
import torch
import matplotlib.pyplot as plt
import json
import time
import shutil
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import random_split, DataLoader, Subset

from Classification.FovalLoss import FocalLoss

class ModelTrainer:
    def __init__(self, dataset, model_class, model_params, training_params, output_dir="TrainingResults"):
        self.dataset = dataset
        self.model_class = model_class
        self.model_params = model_params
        self.training_params = training_params
        
        all_params = {**model_params, **training_params}

        param_summary = "_".join(
            f"{k[:2]}={v:.0e}" if isinstance(v, float) else f"{k[:2]}={v}"
            for k, v in all_params.items()
            if k not in ("input_shape", "num_classes")
        )

        
        #param_summary = f"conv{model_params.get('num_conv_layers', '')}_do{int(model_params.get('use_dropout', False))}_lr{training_params.get('learning_rate', 0):.0e}_bs{training_params.get('batch_size', 0)}"
        self.output_dir = os.path.join(output_dir, param_summary)
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        all_indices = list(range(len(self.dataset)))
        random.shuffle(all_indices)

        train_size = int(self.training_params["train_split"] * len(all_indices))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]
        
        label_map = {i: self.dataset[i][1] for i in train_indices}
        neg_indices = [i for i in train_indices if label_map[i] == 0]
        pos_indices = [i for i in train_indices if label_map[i] == 1]

        desired_ratio = 0.7
        n_pos = len(pos_indices)
        n_neg = int((n_pos / (1 - desired_ratio)) * desired_ratio)
        sampled_neg_indices = random.sample(neg_indices, min(n_neg, len(neg_indices)))

        balanced_train_indices = sampled_neg_indices + pos_indices
        random.shuffle(balanced_train_indices)

        self.train_dataset = Subset(self.dataset, balanced_train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

        #self.train_labels = [self.dataset[i][1] for i in train_indices]
        self.train_labels = [self.dataset[i][1] for i in balanced_train_indices]

        class_sample_count = np.array([self.train_labels.count(t) for t in np.unique(self.train_labels)])

        weight_per_class = 1. / class_sample_count
        samples_weight = np.array([weight_per_class[t] for t in self.train_labels])

        samples_weight = torch.from_numpy(samples_weight).float()
        
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_params["batch_size"],
            sampler=sampler,
            num_workers=3
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_params["batch_size"],
            shuffle=False,
            num_workers=3
        )

        print(f"Počet vzoriek v trénovaní: {len(train_indices)}, počet v validácii: {len(val_indices)}")
        
    def _init_model(self):
        self.model = self.model_class(**self.model_params).to(self.device)
    
        train_labels = self.train_labels #[self.dataset[i][1] for i in self.train_dataset.indices]

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        manual_weights = [0.22, 0.78]
        weights_tensor = torch.tensor(manual_weights, dtype=torch.float).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        #alpha = torch.tensor([0.35, 0.65], dtype=torch.float).to(self.device)
        #self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
       
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_params["learning_rate"], weight_decay=self.training_params["weight_decay"])

    def train(self):
        best_loss = float("inf")
        epochs_no_improve = 0

        train_losses = []
        val_losses = []
        val_accuracies = []
        
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
            
            val_acc = self._evaluate_accuracy()
            val_accuracies.append(val_acc)
            
            cm, f1, precision, recall = self._evaluate_metrics()

            print(f"Epoch [{epoch+1}/{self.training_params['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" 
             f" F1: {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.training_params["patience"]:
                print("Early stopping triggered.")
                break

        self._plot_losses_and_accuracy(train_losses, val_losses, val_accuracies)
        self._final_evaluation()
        
    def _evaluate_metrics(self):
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

        cm = confusion_matrix(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        return cm, f1, precision, recall

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
    
    def _evaluate_accuracy(self):
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
        return accuracy_score(all_labels, all_preds)

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
        
    def _plot_losses_and_accuracy(self, train_losses, val_losses, val_accuracies):
        fig, ax1 = plt.subplots()

        # Left Y-axis (loss)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='tab:blue')
        ax1.plot(train_losses, label='Train Loss', color='tab:orange')
        ax1.plot(val_losses, label='Validation Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)

        # Right Y-axis (accuracy)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Accuracy", color='tab:green')
        ax2.plot(val_accuracies, label='Val Accuracy', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.title("Loss and Accuracy over Epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()


    def _final_evaluation(self):
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)

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

        with open(os.path.join(run_output_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

        with open(os.path.join(run_output_dir, "params.json"), "w") as f:
            json.dump({
                "model_params": self.model_params,
                "training_params": self.training_params
            }, f, indent=4)
            
        shutil.copy(os.path.join(self.output_dir, "best_model.pth"), os.path.join(run_output_dir, "best_model.pth"))
        shutil.copy(os.path.join(self.output_dir, "loss_curve.png"), os.path.join(run_output_dir, "loss_curve.png"))

        os.remove(os.path.join(self.output_dir, "best_model.pth"))
        os.remove(os.path.join(self.output_dir, "loss_curve.png"))

        print("Final Evaluation:")
        print(f"Saved to: {run_output_dir}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
