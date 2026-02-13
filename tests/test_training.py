"""Tests for training functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from molecular_scaffold_aware_multi_task_toxicity_prediction.training.trainer import (
    EarlyStopping,
    ToxicityPredictorTrainer,
)


class TestEarlyStopping:
    """Test cases for EarlyStopping."""

    def test_init_min_mode(self):
        """Test EarlyStopping initialization for minimization."""
        early_stop = EarlyStopping(patience=5, mode='min')
        assert early_stop.patience == 5
        assert early_stop.mode == 'min'
        assert early_stop.best_score == float('inf')
        assert early_stop.counter == 0
        assert early_stop.early_stop is False

    def test_init_max_mode(self):
        """Test EarlyStopping initialization for maximization."""
        early_stop = EarlyStopping(patience=3, mode='max')
        assert early_stop.mode == 'max'
        assert early_stop.best_score == -float('inf')

    def test_improvement_min_mode(self):
        """Test early stopping with improvement (min mode)."""
        early_stop = EarlyStopping(patience=2, mode='min')
        model = nn.Linear(10, 1)

        # First call with high loss
        should_stop = early_stop(1.0, model)
        assert should_stop is False
        assert early_stop.counter == 0
        assert early_stop.best_score == 1.0

        # Second call with lower loss (improvement)
        should_stop = early_stop(0.5, model)
        assert should_stop is False
        assert early_stop.counter == 0
        assert early_stop.best_score == 0.5

    def test_no_improvement_min_mode(self):
        """Test early stopping without improvement (min mode)."""
        early_stop = EarlyStopping(patience=2, mode='min')
        model = nn.Linear(10, 1)

        # Initialize with good score
        early_stop(0.5, model)

        # No improvement
        should_stop = early_stop(0.6, model)
        assert should_stop is False
        assert early_stop.counter == 1

        # Still no improvement - should trigger stop
        should_stop = early_stop(0.7, model)
        assert should_stop is True
        assert early_stop.counter == 2

    def test_improvement_max_mode(self):
        """Test early stopping with improvement (max mode)."""
        early_stop = EarlyStopping(patience=2, mode='max')
        model = nn.Linear(10, 1)

        # First call
        should_stop = early_stop(0.5, model)
        assert should_stop is False
        assert early_stop.best_score == 0.5

        # Improvement
        should_stop = early_stop(0.8, model)
        assert should_stop is False
        assert early_stop.counter == 0
        assert early_stop.best_score == 0.8

    def test_restore_best_weights(self):
        """Test restoring best model weights."""
        early_stop = EarlyStopping(patience=1, restore_best_weights=True)
        model = nn.Linear(10, 1)

        # Get initial weights
        initial_weights = model.state_dict()

        # Call early stopping to save weights
        early_stop(0.5, model)

        # Modify model weights
        with torch.no_grad():
            model.weight.fill_(999.0)

        # Restore best weights
        early_stop.restore_best_model(model)

        # Weights should be restored
        assert not torch.equal(model.weight, torch.full_like(model.weight, 999.0))

    def test_min_delta(self):
        """Test minimum delta for considering improvement."""
        early_stop = EarlyStopping(patience=1, mode='min', min_delta=0.1)
        model = nn.Linear(10, 1)

        # Initialize
        early_stop(1.0, model)

        # Small improvement below min_delta
        should_stop = early_stop(0.95, model)  # Improvement of 0.05 < 0.1
        assert should_stop is False
        assert early_stop.counter == 1  # No improvement recorded

        # Large improvement above min_delta
        should_stop = early_stop(0.8, model)  # Improvement of 0.2 > 0.1
        assert should_stop is False
        assert early_stop.counter == 0  # Improvement recorded


class TestToxicityPredictorTrainer:
    """Test cases for ToxicityPredictorTrainer."""

    @pytest.fixture
    def mock_trainer_setup(self, sample_model, task_names, temp_dir):
        """Setup mock trainer components."""
        # Mock data loaders
        train_loader = MagicMock()
        val_loader = MagicMock()
        test_loader = MagicMock()

        # Mock one batch of data
        batch = MagicMock()
        batch.x = torch.randn(10, 133)
        batch.edge_index = torch.randint(0, 10, (2, 20))
        batch.batch = torch.zeros(10, dtype=torch.long)
        batch.ptr = torch.tensor([0, 10])
        batch.y = torch.randn(1, len(task_names))
        batch.to.return_value = batch

        train_loader.__iter__ = MagicMock(return_value=iter([batch]))
        train_loader.__len__ = MagicMock(return_value=1)
        val_loader.__iter__ = MagicMock(return_value=iter([batch]))
        val_loader.__len__ = MagicMock(return_value=1)
        test_loader.__iter__ = MagicMock(return_value=iter([batch]))

        # Other components
        optimizer = Adam(sample_model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cpu')

        trainer = ToxicityPredictorTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            task_names=task_names,
            checkpoint_dir=temp_dir,
            log_interval=1
        )

        return trainer, batch

    def test_trainer_init(self, mock_trainer_setup):
        """Test trainer initialization."""
        trainer, _ = mock_trainer_setup

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.task_names is not None
        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_score == -float('inf')

    def test_compute_loss(self, mock_trainer_setup):
        """Test loss computation."""
        trainer, _ = mock_trainer_setup

        predictions = torch.randn(4, 3)
        labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float)

        loss = trainer._compute_loss(predictions, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert torch.isfinite(loss).all()

    def test_compute_loss_with_missing_labels(self, mock_trainer_setup):
        """Test loss computation with missing labels."""
        trainer, _ = mock_trainer_setup

        predictions = torch.randn(3, 3)
        labels = torch.tensor([[1, -1, 1], [-1, 0, -1], [0, 1, 0]], dtype=torch.float)

        loss = trainer._compute_loss(predictions, labels)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss).all()

    def test_compute_loss_all_missing(self, mock_trainer_setup):
        """Test loss computation with all missing labels."""
        trainer, _ = mock_trainer_setup

        predictions = torch.randn(2, 3)
        labels = torch.full((2, 3), -1, dtype=torch.float)

        loss = trainer._compute_loss(predictions, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    @patch('src.molecular_scaffold_aware_multi_task_toxicity_prediction.training.trainer.tqdm')
    def test_train_epoch(self, mock_tqdm, mock_trainer_setup):
        """Test training epoch."""
        trainer, batch = mock_trainer_setup

        # Mock progress bar
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value = mock_progress_bar
        mock_progress_bar.__iter__ = MagicMock(return_value=iter([batch]))

        metrics = trainer.train_epoch()

        assert 'train_loss' in metrics
        assert isinstance(metrics['train_loss'], float)
        assert torch.isfinite(torch.tensor(metrics['train_loss']))

        # Check that optimizer was called
        assert trainer.global_step > 0

    @patch('src.molecular_scaffold_aware_multi_task_toxicity_prediction.training.trainer.tqdm')
    def test_validate_epoch(self, mock_tqdm, mock_trainer_setup):
        """Test validation epoch."""
        trainer, batch = mock_trainer_setup

        # Mock progress bar
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value = mock_progress_bar
        mock_progress_bar.__iter__ = MagicMock(return_value=iter([batch]))

        metrics = trainer.validate_epoch()

        assert 'val_loss' in metrics
        assert isinstance(metrics['val_loss'], float)
        assert torch.isfinite(torch.tensor(metrics['val_loss']))

        # Model should be in eval mode during validation
        assert not trainer.model.training

    @patch('src.molecular_scaffold_aware_multi_task_toxicity_prediction.training.trainer.tqdm')
    def test_test_model(self, mock_tqdm, mock_trainer_setup):
        """Test model testing."""
        trainer, batch = mock_trainer_setup

        # Mock progress bar
        mock_tqdm.return_value = iter([batch])

        metrics = trainer.test_model()

        assert len(metrics) > 0
        assert any(k.startswith('test_') for k in metrics.keys())

    def test_save_checkpoint(self, mock_trainer_setup, temp_dir):
        """Test checkpoint saving."""
        trainer, _ = mock_trainer_setup

        checkpoint_path = temp_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'best_val_score' in checkpoint
        assert 'training_history' in checkpoint

    def test_load_checkpoint(self, mock_trainer_setup, temp_dir):
        """Test checkpoint loading."""
        trainer, _ = mock_trainer_setup

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        trainer.epoch = 5
        trainer.best_val_score = 0.85
        trainer.save_checkpoint(checkpoint_path)

        # Reset trainer state
        trainer.epoch = 0
        trainer.best_val_score = -float('inf')

        # Load checkpoint
        loaded_checkpoint = trainer.load_checkpoint(checkpoint_path)

        assert trainer.epoch == 5
        assert trainer.best_val_score == 0.85
        assert isinstance(loaded_checkpoint, dict)

    def test_log_model_info(self, mock_trainer_setup):
        """Test model info logging."""
        trainer, _ = mock_trainer_setup

        # Should not raise any exceptions
        with patch('mlflow.log_params') as mock_log_params:
            trainer._log_model_info()
            assert mock_log_params.called

    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    @patch('src.molecular_scaffold_aware_multi_task_toxicity_prediction.training.trainer.tqdm')
    def test_train_basic(self, mock_tqdm, mock_set_exp, mock_start_run, mock_trainer_setup):
        """Test basic training loop."""
        trainer, batch = mock_trainer_setup

        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_123'
        mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Mock progress bars
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value = mock_progress_bar
        mock_progress_bar.__iter__ = MagicMock(return_value=iter([batch]))

        # Train for 1 epoch
        history = trainer.train(num_epochs=1, mlflow_experiment="test_exp")

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 1

    def test_early_stopping_integration(self, mock_trainer_setup):
        """Test early stopping integration."""
        trainer, _ = mock_trainer_setup

        # Add early stopping
        trainer.early_stopping = EarlyStopping(patience=1, mode='max')

        # Should not raise any exceptions
        assert trainer.early_stopping is not None
        assert trainer.early_stopping.patience == 1

    @pytest.mark.parametrize('scheduler_type', [None, 'plateau'])
    def test_scheduler_integration(self, scheduler_type, mock_trainer_setup):
        """Test learning rate scheduler integration."""
        trainer, _ = mock_trainer_setup

        if scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            trainer.scheduler = ReduceLROnPlateau(trainer.optimizer)
        else:
            trainer.scheduler = None

        # Should not raise any exceptions during validation
        metrics = {}
        if scheduler_type == 'plateau':
            metrics['val_loss'] = 0.5

        # Test that scheduler step can be called
        if trainer.scheduler is not None:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(metrics.get('val_loss', 0.5))
            else:
                trainer.scheduler.step()

    def test_device_handling(self, sample_model, task_names, temp_dir):
        """Test device handling in trainer."""
        # Test CPU device
        device = torch.device('cpu')

        train_loader = MagicMock()
        val_loader = MagicMock()
        test_loader = MagicMock()
        optimizer = Adam(sample_model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        trainer = ToxicityPredictorTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            task_names=task_names,
            checkpoint_dir=temp_dir
        )

        assert trainer.device == device

        # Test CUDA device (if available)
        if torch.cuda.is_available():
            device_cuda = torch.device('cuda')
            trainer_cuda = ToxicityPredictorTrainer(
                model=sample_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device_cuda,
                task_names=task_names,
                checkpoint_dir=temp_dir
            )
            assert trainer_cuda.device == device_cuda