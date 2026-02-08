#!/usr/bin/env python3
"""Training script for molecular toxicity prediction models.

This script handles the complete training pipeline including data loading,
model initialization, training with MLflow tracking, and final evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molecular_scaffold_aware_multi_task_toxicity_prediction.data.loader import (
    MoleculeNetLoader,
    ScaffoldSplitter,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.data.preprocessing import (
    GraphFeaturizer,
    MoleculePreprocessor,
    ScaffoldAwareTransform,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.models.model import (
    MultiTaskToxicityPredictor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.training.trainer import (
    EarlyStopping,
    ToxicityPredictorTrainer,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.utils.config import (
    Config,
    get_device,
    get_model_config,
    load_config,
    set_random_seeds,
    setup_logging,
    validate_config,
)

logger = logging.getLogger(__name__)


class ToxicityDataset:
    """Custom dataset for molecular toxicity prediction."""

    def __init__(self, graphs: List[Data]):
        """Initialize dataset.

        Args:
            graphs: List of graph data objects
        """
        self.graphs = graphs

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Get graph by index."""
        return self.graphs[idx]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train molecular toxicity prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default='./data',
        help='Directory to store/load datasets'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default='./outputs',
        help='Output directory for checkpoints and results'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        help='MLflow experiment name (overrides config)'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        help='MLflow run name'
    )

    parser.add_argument(
        '--resume',
        type=Path,
        help='Path to checkpoint to resume training'
    )

    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation (requires --resume)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for training'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )

    return parser.parse_args()


def load_and_preprocess_data(config: Config, data_dir: Path) -> Tuple[Dict, List[str]]:
    """Load and preprocess molecular data.

    Args:
        config: Configuration object
        data_dir: Data directory path

    Returns:
        Tuple of (datasets_dict, task_names)

    Raises:
        ValueError: If dataset configuration is invalid
        FileNotFoundError: If data directory doesn't exist
        RuntimeError: If data loading or processing fails
    """
    try:
        logger.info("Loading molecular data...")

        # Validate data directory
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {data_dir}")

        # Initialize data loader
        try:
            loader = MoleculeNetLoader(data_dir=data_dir, cache=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MoleculeNetLoader: {e}") from e

        dataset_name = config.get('data.dataset', required=True)
        if not dataset_name:
            raise ValueError("Dataset name not specified in configuration")

        logger.info(f"Loading dataset: {dataset_name}")

        # Load dataset
        try:
            df = loader.load_dataset(dataset_name)
            task_info = loader.get_task_info(dataset_name)
            task_names = task_info['task_names']
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}") from e

        if len(df) == 0:
            raise RuntimeError(f"Dataset '{dataset_name}' is empty")

        if len(task_names) == 0:
            raise RuntimeError(f"No tasks found for dataset '{dataset_name}'")

        logger.info(f"Loaded {len(df)} molecules with {len(task_names)} toxicity tasks")

        # Initialize preprocessor and featurizer
        preprocessor = MoleculePreprocessor(
            remove_salts=config.get('data.remove_salts', True),
            canonical_smiles=config.get('data.canonical_smiles', True),
            remove_stereochemistry=config.get('data.remove_stereochemistry', False)
        )

        featurizer = GraphFeaturizer(
            explicit_h=config.get('data.explicit_h', False),
            use_chirality=config.get('data.use_chirality', True)
        )

        # Optional scaffold-aware transform
        scaffold_transform = None
        if config.get('data.use_scaffold_transform', False):
            scaffold_transform = ScaffoldAwareTransform(
                scaffold_type=config.get('data.scaffold_type', 'murcko'),
                embedding_dim=config.get('model.scaffold_dim', 64)
            )

        # Preprocess and featurize molecules
        logger.info("Preprocessing molecules and creating graphs...")
        graphs = []
        valid_indices = []

        for idx, row in df.iterrows():
            smiles = row['smiles']

            # Preprocess SMILES
            processed_smiles = preprocessor.preprocess_smiles(smiles)
            if processed_smiles is None:
                continue

            # Extract labels
            labels = []
            for task in task_names:
                label = row[task]
                if pd.isna(label):
                    labels.append(-1.0)  # Missing label indicator
                else:
                    labels.append(float(label))

            # Create graph
            graph = featurizer.featurize(processed_smiles, labels)
            if graph is None:
                continue

            # Apply scaffold transform if enabled
            if scaffold_transform is not None:
                graph = scaffold_transform(graph)

            graphs.append(graph)
            valid_indices.append(idx)

        logger.info(f"Successfully processed {len(graphs)} molecules")

        # Split data based on scaffolds or random
        if config.get('data.scaffold_split', True):
            logger.info("Performing scaffold-based splitting...")
            splitter = ScaffoldSplitter(
                scaffold_func=config.get('data.scaffold_func', 'murcko'),
                random_state=config.get('training.random_seed', 42)
            )

            valid_smiles = [df.iloc[idx]['smiles'] for idx in valid_indices]
            train_idx, val_idx, test_idx = splitter.split(
                valid_smiles,
                train_ratio=config.get('data.train_ratio', 0.8),
                val_ratio=config.get('data.val_ratio', 0.1),
                test_ratio=config.get('data.test_ratio', 0.1)
            )

            # Analyze scaffold diversity
            scaffold_stats = splitter.analyze_scaffold_diversity(valid_smiles)
            logger.info(f"Scaffold diversity: {scaffold_stats['unique_scaffolds']} unique scaffolds "
                       f"({scaffold_stats['scaffold_ratio']:.3f} ratio)")

        else:
            logger.info("Performing random splitting...")
            from sklearn.model_selection import train_test_split

            indices = list(range(len(graphs)))
            train_idx, temp_idx = train_test_split(
                indices,
                train_size=config.get('data.train_ratio', 0.8),
                random_state=config.get('training.random_seed', 42)
            )

            val_ratio = config.get('data.val_ratio', 0.1)
            test_ratio = config.get('data.test_ratio', 0.1)
            val_size = val_ratio / (val_ratio + test_ratio)

            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                random_state=config.get('training.random_seed', 42)
            )

        # Create datasets
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        datasets = {
            'train': ToxicityDataset(train_graphs),
            'val': ToxicityDataset(val_graphs),
            'test': ToxicityDataset(test_graphs)
        }

        logger.info(f"Dataset splits: train={len(train_graphs)}, "
                   f"val={len(val_graphs)}, test={len(test_graphs)}")

        return datasets, task_names

    except Exception as e:
        logger.error(f"Failed to load and preprocess data: {e}")
        raise


def create_data_loaders(datasets: Dict, config: Config, num_workers: int) -> Dict[str, DataLoader]:
    """Create PyTorch data loaders.

    Args:
        datasets: Dictionary of datasets
        config: Configuration object
        num_workers: Number of data loader workers

    Returns:
        Dictionary of data loaders
    """
    batch_size = config.get('training.batch_size', 32)

    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }

    return loaders


def create_model(config: Config, task_names: List[str], device: torch.device) -> MultiTaskToxicityPredictor:
    """Create and initialize model.

    Args:
        config: Configuration object
        task_names: List of task names
        device: Training device

    Returns:
        Initialized model
    """
    model_name = config.get('model.name', required=True)
    backbone_config = get_model_config(config, model_name)

    model = MultiTaskToxicityPredictor(
        backbone=model_name,
        backbone_config=backbone_config,
        num_tasks=len(task_names),
        hidden_dims=config.get('model.prediction_hidden_dims', [128, 64]),
        dropout=config.get('model.prediction_dropout', 0.3),
        use_task_embedding=config.get('model.use_task_embedding', True)
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Created {model_name.upper()} model with {total_params:,} parameters "
               f"({trainable_params:,} trainable)")

    return model


def create_optimizer_and_scheduler(model: nn.Module, config: Config):
    """Create optimizer and learning rate scheduler.

    Args:
        model: PyTorch model
        config: Configuration object

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_name = config.get('training.optimizer', 'adamw').lower()
    learning_rate = config.get('training.learning_rate', 1e-3)
    weight_decay = config.get('training.weight_decay', 1e-5)

    if optimizer_name == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    scheduler = None
    scheduler_type = config.get('training.scheduler', None)

    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Monitor AUC (higher is better)
            factor=config.get('training.scheduler_factor', 0.5),
            patience=config.get('training.scheduler_patience', 5),
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config.get('training.scheduler_step_size', 10),
            gamma=config.get('training.scheduler_gamma', 0.9)
        )

    logger.info(f"Created {optimizer_name} optimizer with lr={learning_rate}")
    if scheduler:
        logger.info(f"Created {scheduler_type} scheduler")

    return optimizer, scheduler


def main():
    """Orchestrate complete molecular toxicity prediction model training pipeline.

    Executes the full training workflow including configuration loading,
    data preprocessing, model creation, training loop with validation,
    early stopping, checkpoint saving, and MLflow experiment tracking.

    The pipeline stages:
    1. Parse arguments and load configuration
    2. Set up logging, device, and random seeds
    3. Load and preprocess molecular datasets
    4. Create model, optimizer, and scheduler
    5. Initialize trainer and execute training
    6. Save final model checkpoint

    Raises:
        SystemExit: On critical errors that prevent training (config, data, or model failures)
    """
    args = parse_args()

    try:
        # Setup logging
        log_level = 'DEBUG' if args.debug else 'INFO'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Load configuration with error handling
        logger.info(f"Loading configuration from {args.config}")
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

        # Override config with command line arguments
        if args.device != 'auto':
            config['device'] = args.device

        # Validate configuration
        try:
            validate_config(config)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)

        # Setup
        try:
            set_random_seeds(config.get('training.random_seed', 42))
            device = get_device(config)
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            sys.exit(1)

        # Create output directory
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            sys.exit(1)

        # Load and preprocess data
        try:
            datasets, task_names = load_and_preprocess_data(config, args.data_dir)
            data_loaders = create_data_loaders(datasets, config, args.num_workers)
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            sys.exit(1)

        # Create model
        try:
            model = create_model(config, task_names, device)
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            sys.exit(1)

        # Create optimizer and scheduler
        try:
            optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        except Exception as e:
            logger.error(f"Optimizer/scheduler creation failed: {e}")
            sys.exit(1)

        # Create loss function
        try:
            criterion = nn.BCEWithLogitsLoss()
        except Exception as e:
            logger.error(f"Loss function creation failed: {e}")
            sys.exit(1)

        # Create early stopping
        early_stopping = None
        if config.get('training.early_stopping', True):
            try:
                early_stopping = EarlyStopping(
                    patience=config.get('training.patience', 10),
                    min_delta=config.get('training.min_delta', 1e-4),
                    mode='max',  # Monitor validation AUC
                    restore_best_weights=True
                )
            except Exception as e:
                logger.warning(f"Early stopping setup failed, continuing without it: {e}")
                early_stopping = None

        # Create trainer
        try:
            trainer = ToxicityPredictorTrainer(
                model=model,
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val'],
                test_loader=data_loaders['test'],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task_names=task_names,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_dir=args.output_dir / 'checkpoints',
                log_interval=config.get('training.log_interval', 10)
            )
        except Exception as e:
            logger.error(f"Trainer creation failed: {e}")
            sys.exit(1)

        # Resume from checkpoint if specified
        if args.resume:
            try:
                logger.info(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                sys.exit(1)

        # Run evaluation only
        if args.eval_only:
            if not args.resume:
                logger.error("--eval-only requires --resume")
                sys.exit(1)

            try:
                logger.info("Running evaluation only...")
                test_metrics = trainer.test_model()

                # Print results
                print("\nTest Results:")
                print("=" * 50)
                for metric, value in test_metrics.items():
                    if 'auc_roc' in metric:
                        print(f"{metric}: {value:.4f}")

                return
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                sys.exit(1)

        # Training
        experiment_name = args.experiment_name or config.get('mlflow.experiment_name', 'toxicity_prediction')
        run_name = args.run_name or config.get('mlflow.run_name', None)
        num_epochs = config.get('training.num_epochs', 100)

        logger.info(f"Starting training for {num_epochs} epochs...")

        try:
            training_history = trainer.train(
                num_epochs=num_epochs,
                mlflow_experiment=experiment_name,
                run_name=run_name
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Save emergency checkpoint
            try:
                emergency_path = args.output_dir / 'emergency_checkpoint.pt'
                trainer.save_checkpoint(emergency_path)
                logger.info(f"Saved emergency checkpoint to {emergency_path}")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
            sys.exit(1)

        # Save final model
        try:
            final_model_path = args.output_dir / 'final_model.pt'
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save final model: {e}")

        # Save training history
        try:
            import json
            history_path = args.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            logger.info(f"Saved training history to {history_path}")
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()