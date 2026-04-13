"""
Training and validation routines for deep learning models.
"""

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


class Trainer:
    """Trainer class for handling model training and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        writer: SummaryWriter = None,
        scheduler=None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device (cuda or cpu)
            writer: TensorBoard writer
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.scheduler = scheduler
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average epoch loss
        """
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Log batch-level loss
            if self.writer:
                self.writer.add_scalar(
                    "Loss/train_batch",
                    loss.item(),
                    self.global_step
                )
            
            self.global_step += 1
        
        epoch_loss = running_loss / len(self.train_loader)
        
        # Log epoch-level loss
        if self.writer:
            self.writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
            self.writer.add_scalar(
                "LearningRate",
                self.optimizer.param_groups[0]["lr"],
                epoch
            )
        
        return epoch_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (validation_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        # Log validation metrics
        if self.writer:
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", accuracy, epoch)
        
        return val_loss, accuracy
    
    def log_weights_histograms(self, epoch: int):
        """Log weight histograms to TensorBoard."""
        if not self.writer:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
    
    def log_sample_images(self, epoch: int, num_images: int = 8):
        """Log sample images to TensorBoard."""
        if not self.writer:
            return
        
        self.model.eval()
        data_iter = iter(self.val_loader)
        images, labels = next(data_iter)
        
        img_grid = vutils.make_grid(images[:num_images])
        self.writer.add_image("Sample Inputs", img_grid, epoch)
    
    def train(self, num_epochs: int, checkpoint_dir: str = None) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_times": []
        }
        
        best_val_accuracy = 0.0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_loss, val_accuracy = self.validate(epoch)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            
            # Log histograms and images
            self.log_weights_histograms(epoch)
            self.log_sample_images(epoch)
            
            epoch_time = time.time() - epoch_start_time
            history["epoch_times"].append(epoch_time)
            
            # Update learning rate scheduler
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step(val_loss)
            
            # Print progress
            print(
                f"Epoch {epoch+1:2d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % 5 == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch)
            
            # Track best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                if checkpoint_dir:
                    best_checkpoint_path = f"{checkpoint_dir}/best_model.pt"
                    self.save_checkpoint(best_checkpoint_path, epoch, is_best=True)
        
        print(f"\nBest validation accuracy: {best_val_accuracy:.2f}% (epoch {best_epoch+1})")
        
        return history
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "is_best": is_best
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.0,
    device: str = "cuda",
    writer: SummaryWriter = None,
    scheduler_name: str = None,
    checkpoint_dir: str = None
) -> Dict:
    """
    Convenience function to train a model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        optimizer_name: Optimizer name (Adam, SGD)
        weight_decay: L2 regularization weight
        device: Device
        writer: TensorBoard writer
        scheduler_name: Learning rate scheduler name
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Training history dictionary
    """
    
    # Create optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create scheduler if specified
    scheduler = None
    if scheduler_name:
        if scheduler_name.lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        elif scheduler_name.lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
    
    # Create loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        writer=writer,
        scheduler=scheduler
    )
    
    # Train
    history = trainer.train(num_epochs, checkpoint_dir=checkpoint_dir)
    
    return history
