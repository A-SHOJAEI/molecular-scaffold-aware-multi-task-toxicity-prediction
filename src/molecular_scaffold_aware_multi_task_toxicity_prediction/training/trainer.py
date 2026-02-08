"""Training utilities for molecular toxicity prediction models."""

import logging
import os
import time
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from ..evaluation.metrics import ToxicityMetrics, MultiTaskEvaluator

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like AUC
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf

    def __call__(self,
                 current_score: float,
                 model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            current_score: Current validation score
            model: Model to save weights from

        Returns:
            True if training should stop
        """
        if self.monitor_op(current_score, self.best_score - self.min_delta):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_model(self, model: nn.Module) -> None:
        """Restore best model weights.

        Args:
            model: Model to restore weights to
        """
        if self.best_weights is not None and self.restore_best_weights:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class ToxicityPredictorTrainer:
    """Trainer for molecular toxicity prediction models.

    Handles training loop, validation, checkpointing, and MLflow tracking
    for multi-task toxicity prediction experiments.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 task_names: List[str],
                 scheduler: Optional[_LRScheduler] = None,
                 early_stopping: Optional[EarlyStopping] = None,
                 checkpoint_dir: Union[str, Path] = "./checkpoints",
                 log_interval: int = 10):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer for training
            criterion: Loss function
            device: Training device
            task_names: Names of prediction tasks
            scheduler: Learning rate scheduler
            early_stopping: Early stopping configuration
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval in batches
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.task_names = task_names
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.log_interval = log_interval

        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.toxicity_metrics = ToxicityMetrics(task_names=task_names)
        self.evaluator = MultiTaskEvaluator(
            task_names=task_names,
            metrics=['auc_roc', 'auc_pr', 'accuracy', 'f1']
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_score = -np.inf
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'lr': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics

        Raises:
            RuntimeError: If training fails due to model or data issues
            ValueError: If invalid data is encountered
            torch.cuda.OutOfMemoryError: If GPU memory is exhausted
        """
        try:
            # Performance tracking
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

            logger.debug(f"Starting training epoch with {len(self.train_loader)} batches")
            logger.debug(f"Initial memory usage: {start_memory:.1f} MB")

            self.model.train()
            total_loss = 0.0
            num_batches = 0
            predictions_list = []
            labels_list = []

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch + 1} [Train]",
                leave=False
            )

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device with error handling
                    if batch is None:
                        logger.warning(f"Skipping None batch at index {batch_idx}")
                        continue

                    batch = batch.to(self.device)
                    labels = batch.y  # [batch_size, num_tasks]

                    # Validate batch data
                    if labels.numel() == 0:
                        logger.warning(f"Skipping empty batch at index {batch_idx}")
                        continue

                    # Forward pass
                    self.optimizer.zero_grad()
                    predictions = self.model(batch)  # [batch_size, num_tasks]

                    # Compute loss (handle missing labels)
                    loss = self._compute_loss(predictions, labels)

                    # Check for invalid loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"Invalid loss detected at batch {batch_idx}: {loss.item()}")
                        continue

                    # Backward pass
                    loss.backward()

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    self.global_step += 1

                    # Store for epoch-level metrics
                    predictions_list.append(predictions.detach().cpu())
                    labels_list.append(labels.detach().cpu())

                    # Logging
                    if batch_idx % self.log_interval == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                            'lr': f"{current_lr:.2e}"
                        })

                        # MLflow logging
                        try:
                            if mlflow.active_run():
                                mlflow.log_metrics({
                                    'train_loss_step': loss.item(),
                                    'learning_rate': current_lr
                                }, step=self.global_step)
                        except Exception as e:
                            logger.warning(f"Failed to log metrics to MLflow: {e}")

                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU out of memory at batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    raise
                except (RuntimeError, ValueError) as e:
                    logger.error(f"Model computation error at batch {batch_idx}: {e}")
                    continue
                except (AttributeError, TypeError) as e:
                    logger.error(f"Data format error at batch {batch_idx}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing batch {batch_idx}: {e}")
                    continue

            if num_batches == 0:
                raise RuntimeError("No valid batches processed during training epoch")

            # Compute epoch metrics
            avg_loss = total_loss / num_batches
            predictions_epoch = torch.cat(predictions_list, dim=0)
            labels_epoch = torch.cat(labels_list, dim=0)

            # Convert to probabilities for metrics
            probabilities = torch.sigmoid(predictions_epoch)

            # Compute task-specific metrics with error handling
            try:
                task_metrics = self.toxicity_metrics.compute_metrics(
                    probabilities.numpy(),
                    labels_epoch.numpy()
                )
            except Exception as e:
                logger.error(f"Error computing task metrics: {e}")
                task_metrics = {}

            # Performance metrics
            end_time = time.time()
            epoch_duration = end_time - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory

            # Log performance metrics
            logger.info(f"Training epoch completed in {epoch_duration:.2f}s")
            logger.debug(f"Memory usage: {start_memory:.1f} -> {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            logger.debug(f"Processed {num_batches} batches ({num_batches/epoch_duration:.1f} batches/s)")

            if torch.cuda.is_available():
                peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                logger.debug(f"GPU memory: {current_gpu_memory:.1f} MB (peak: {peak_gpu_memory:.1f} MB)")

            metrics = {
                'train_loss': avg_loss,
                'train_duration': epoch_duration,
                'train_memory_mb': end_memory,
                **{f'train_{k}': v for k, v in task_metrics.items()}
            }

            return metrics

        except Exception as e:
            logger.error(f"Training epoch failed: {e}")
            raise RuntimeError(f"Training epoch failed: {e}") from e

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.

        Returns:
            Dictionary with validation metrics

        Raises:
            RuntimeError: If validation fails due to model or data issues
            ValueError: If invalid data is encountered
        """
        try:
            self.model.eval()
            total_loss = 0.0
            num_batches = 0
            predictions_list = []
            labels_list = []

            with torch.no_grad():
                progress_bar = tqdm(
                    self.val_loader,
                    desc=f"Epoch {self.epoch + 1} [Val]",
                    leave=False
                )

                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        if batch is None:
                            logger.warning(f"Skipping None validation batch at index {batch_idx}")
                            continue

                        batch = batch.to(self.device)
                        labels = batch.y

                        # Validate batch data
                        if labels.numel() == 0:
                            logger.warning(f"Skipping empty validation batch at index {batch_idx}")
                            continue

                        # Forward pass
                        predictions = self.model(batch)
                        loss = self._compute_loss(predictions, labels)

                        # Check for invalid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"Invalid validation loss at batch {batch_idx}: {loss.item()}")
                            continue

                        # Update metrics
                        total_loss += loss.item()
                        num_batches += 1

                        # Store for epoch-level metrics
                        predictions_list.append(predictions.cpu())
                        labels_list.append(labels.cpu())

                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'avg_loss': f"{total_loss / num_batches:.4f}"
                        })

                    except Exception as e:
                        logger.error(f"Error processing validation batch {batch_idx}: {e}")
                        continue

            if num_batches == 0:
                raise RuntimeError("No valid batches processed during validation")

            # Compute epoch metrics
            avg_loss = total_loss / num_batches
            predictions_epoch = torch.cat(predictions_list, dim=0)
            labels_epoch = torch.cat(labels_list, dim=0)

            # Convert to probabilities
            probabilities = torch.sigmoid(predictions_epoch)

            # Compute comprehensive metrics with error handling
            try:
                task_metrics = self.toxicity_metrics.compute_metrics(
                    probabilities.numpy(),
                    labels_epoch.numpy()
                )
            except Exception as e:
                logger.error(f"Error computing task metrics: {e}")
                task_metrics = {}

            metrics = {
                'val_loss': avg_loss,
                **{f'val_{k}': v for k, v in task_metrics.items()},
            }

            return metrics

        except Exception as e:
            logger.error(f"Validation epoch failed: {e}")
            raise RuntimeError(f"Validation epoch failed: {e}") from e

    def test_model(self) -> Dict[str, float]:
        """Evaluate model on test set.

        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        predictions_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.device)
                predictions = self.model(batch)

                predictions_list.append(predictions.cpu())
                labels_list.append(batch.y.cpu())

        # Compute test metrics
        predictions_all = torch.cat(predictions_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        probabilities = torch.sigmoid(predictions_all)

        # Task-specific metrics
        task_metrics = self.toxicity_metrics.compute_metrics(
            probabilities.numpy(),
            labels_all.numpy()
        )

        # Multi-task metrics
        multi_task_metrics = self.evaluator.evaluate(
            probabilities.numpy(),
            labels_all.numpy()
        )

        test_metrics = {
            **{f'test_{k}': v for k, v in task_metrics.items()},
            **{f'test_{k}': v for k, v in multi_task_metrics.items()}
        }

        return test_metrics

    def train(self,
              num_epochs: int,
              mlflow_experiment: Optional[str] = None,
              run_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Train the model for specified epochs.

        Args:
            num_epochs: Number of epochs to train
            mlflow_experiment: MLflow experiment name
            mlflow_run_name: MLflow run name

        Returns:
            Training history dictionary
        """
        # Setup MLflow
        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=run_name) as run:
            # Log model parameters
            self._log_model_info()

            logger.info(f"Starting training for {num_epochs} epochs")
            logger.info(f"MLflow run ID: {run.info.run_id}")

            for epoch in range(num_epochs):
                self.epoch = epoch
                epoch_start_time = time.time()

                # Train and validate
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()

                # Compute current validation score
                current_val_score = val_metrics.get('val_auc_roc_mean', 0)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_val_score)
                    else:
                        self.scheduler.step()

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                epoch_time = time.time() - epoch_start_time

                # Update training history
                self.training_history['train_loss'].append(train_metrics['train_loss'])
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_metrics'].append(current_val_score)
                self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
                if current_val_score > self.best_val_score:
                    self.best_val_score = current_val_score
                    self.save_checkpoint(is_best=True)

                # MLflow logging
                epoch_metrics['epoch_time'] = epoch_time
                # Filter out non-numeric values for MLflow
                numeric_metrics = {k: v for k, v in epoch_metrics.items()
                                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
                try:
                    mlflow.log_metrics(numeric_metrics, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log epoch metrics to MLflow: {e}")

                # Progress logging
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val AUC: {current_val_score:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                # Early stopping
                if self.early_stopping:
                    stop_metric = val_metrics['val_loss'] if self.early_stopping.mode == 'min' else current_val_score
                    if self.early_stopping(stop_metric, self.model):
                        logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                        break

            # Restore best model if early stopping
            if self.early_stopping and self.early_stopping.restore_best_weights:
                self.early_stopping.restore_best_model(self.model)

            # Final test evaluation
            test_metrics = self.test_model()
            try:
                mlflow.log_metrics(test_metrics, step=num_epochs)
            except Exception as e:
                logger.warning(f"Failed to log test metrics to MLflow: {e}")

            # Log model
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    name="model",
                )
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")

            logger.info("Training completed!")
            logger.info(f"Best validation AUC: {self.best_val_score:.4f}")

            # Print final test results
            for metric, value in test_metrics.items():
                if 'auc_roc' in metric:
                    logger.info(f"{metric}: {value:.4f}")

        return self.training_history

    def _compute_loss(self,
                      predictions: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Compute loss handling missing labels.

        Args:
            predictions: Model predictions [batch_size, num_tasks]
            labels: True labels [batch_size, num_tasks]

        Returns:
            Computed loss
        """
        # Handle missing labels (typically -1 or NaN)
        valid_mask = (labels != -1) & (~torch.isnan(labels))

        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Apply loss only to valid labels
        valid_predictions = predictions[valid_mask]
        valid_labels = labels[valid_mask]

        return self.criterion(valid_predictions, valid_labels)

    def _log_model_info(self) -> None:
        """Log comprehensive model and training configuration to MLflow.

        Logs model parameters (total/trainable counts), architecture details,
        task configuration, optimizer settings, and device information for
        experiment tracking and reproducibility.

        Note:
            This method assumes an active MLflow run context.
        """
        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        mlflow.log_params({
            'model_name': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_tasks': len(self.task_names),
            'task_names': ','.join(self.task_names),
            'optimizer': self.optimizer.__class__.__name__,
            'criterion': self.criterion.__class__.__name__,
            'device': str(self.device),
        })

        # Training configuration
        mlflow.log_params({
            'train_size': len(self.train_loader.dataset),
            'val_size': len(self.val_loader.dataset),
            'test_size': len(self.test_loader.dataset),
            'batch_size': self.train_loader.batch_size,
        })

    def save_checkpoint(self,
                       checkpoint_path: Optional[Union[str, Path]] = None,
                       is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            is_best: Whether this is the best checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch + 1}.pt"

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'training_history': self.training_history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        try:
            torch.save(checkpoint, checkpoint_path)
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"PyTorch save error for {checkpoint_path}: {e}")
            raise

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            try:
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")
            except (OSError, IOError, PermissionError) as e:
                logger.error(f"Failed to save best model to {best_path}: {e}")
                # Don't raise here as main checkpoint was saved successfully

    def load_checkpoint(self,
                       checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
            KeyError: If required keys are missing from checkpoint
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            logger.info(f"Loading checkpoint from: {checkpoint_path}")

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint file: {e}") from e

            # Validate required checkpoint keys
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")

            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                raise RuntimeError(f"Failed to load model state dict: {e}") from e

            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state dict: {e}")

            # Load optional fields with defaults
            self.epoch = checkpoint.get('epoch', 0)
            self.best_val_score = checkpoint.get('best_val_score', -np.inf)
            self.training_history = checkpoint.get('training_history', {
                'train_loss': [],
                'val_loss': [],
                'val_metrics': [],
                'lr': []
            })

            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state dict: {e}")

            logger.info(f"Successfully loaded checkpoint from epoch {self.epoch + 1}")
            logger.info(f"Best validation score from checkpoint: {self.best_val_score:.4f}")

            return checkpoint

        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            raise