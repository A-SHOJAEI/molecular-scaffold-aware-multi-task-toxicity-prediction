#!/usr/bin/env python3
"""Evaluation script for molecular toxicity prediction models.

This script performs comprehensive evaluation including scaffold generalization
analysis, visualization of results, and comparison between models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molecular_scaffold_aware_multi_task_toxicity_prediction.data.loader import (
    MoleculeNetLoader,
    ScaffoldSplitter,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.data.preprocessing import (
    GraphFeaturizer,
    MoleculePreprocessor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.evaluation.metrics import (
    MultiTaskEvaluator,
    ScaffoldGeneralizationAnalyzer,
    ToxicityMetrics,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.models.model import (
    MultiTaskToxicityPredictor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.utils.config import (
    Config,
    get_device,
    get_model_config,
    load_config,
    set_random_seeds,
)

logger = logging.getLogger(__name__)


class ToxicityDataset:
    """Custom dataset for molecular toxicity prediction."""

    def __init__(self, graphs: List, smiles: List[str], scaffolds: Optional[List[str]] = None):
        """Initialize dataset.

        Args:
            graphs: List of graph data objects
            smiles: List of SMILES strings
            scaffolds: Optional list of scaffold identifiers
        """
        self.graphs = graphs
        self.smiles = smiles
        self.scaffolds = scaffolds or [''] * len(graphs)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[Data, str, str]:
        """Get graph, SMILES, and scaffold by index.

        Returns:
            Tuple containing (graph_data, smiles_string, scaffold_string).
        """
        return self.graphs[idx], self.smiles[idx], self.scaffolds[idx]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate molecular toxicity prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default='./data',
        help='Directory containing datasets'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default='./evaluation_results',
        help='Output directory for evaluation results'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['tox21', 'toxcast', 'clintox'],
        help='Dataset to evaluate (overrides config)'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='test',
        help='Data split to evaluate'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--scaffold-analysis',
        action='store_true',
        help='Perform scaffold generalization analysis'
    )

    parser.add_argument(
        '--compare-with',
        type=Path,
        help='Path to another checkpoint for comparison'
    )

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )

    parser.add_argument(
        '--plot-results',
        action='store_true',
        help='Generate result plots'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for evaluation'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def load_model_and_data(config: Config,
                       checkpoint_path: Path,
                       data_dir: Path,
                       dataset_name: Optional[str] = None) -> Tuple:
    """Load model and data for evaluation.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        data_dir: Data directory
        dataset_name: Dataset name (overrides config)

    Returns:
        Tuple[torch.nn.Module, Dict[str, pd.DataFrame], List[str], torch.device, np.ndarray]:
            - model: Trained multi-task toxicity prediction model
            - datasets: Dictionary with 'train', 'val', 'test' dataset splits
            - task_names: List of toxicity endpoint names
            - device: PyTorch device (CPU/CUDA)
            - train_idx: Training set indices for scaffold analysis
    """
    # Determine dataset
    dataset = dataset_name or config.get('data.dataset', required=True)
    device = get_device(config)

    logger.info(f"Loading data for dataset: {dataset}")

    # Load dataset
    loader = MoleculeNetLoader(data_dir=data_dir, cache=True)
    df = loader.load_dataset(dataset)
    task_info = loader.get_task_info(dataset)
    task_names = task_info['task_names']

    logger.info(f"Dataset contains {len(df)} molecules with {len(task_names)} tasks")

    # Initialize preprocessor and featurizer
    preprocessor = MoleculePreprocessor(
        remove_salts=config.get('data.remove_salts', True),
        canonical_smiles=config.get('data.canonical_smiles', True)
    )

    featurizer = GraphFeaturizer(
        explicit_h=config.get('data.explicit_h', False),
        use_chirality=config.get('data.use_chirality', True)
    )

    # Process molecules
    graphs = []
    valid_smiles = []
    valid_scaffolds = []
    valid_labels = []

    # Initialize scaffold splitter for scaffold computation
    splitter = ScaffoldSplitter()

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
            labels.append(float(label) if not pd.isna(label) else -1.0)

        # Create graph
        graph = featurizer.featurize(processed_smiles, labels)
        if graph is None:
            continue

        graphs.append(graph)
        valid_smiles.append(processed_smiles)
        valid_labels.append(labels)

    # Compute scaffolds
    all_scaffolds = splitter._generate_scaffolds(valid_smiles)
    valid_scaffolds = all_scaffolds

    logger.info(f"Successfully processed {len(graphs)} molecules")

    # Split data (same logic as training)
    if config.get('data.scaffold_split', True):
        train_idx, val_idx, test_idx = splitter.split(
            valid_smiles,
            train_ratio=config.get('data.train_ratio', 0.8),
            val_ratio=config.get('data.val_ratio', 0.1),
            test_ratio=config.get('data.test_ratio', 0.1)
        )
    else:
        from sklearn.model_selection import train_test_split
        indices = list(range(len(graphs)))
        train_idx, temp_idx = train_test_split(
            indices, train_size=config.get('data.train_ratio', 0.8),
            random_state=config.get('training.random_seed', 42)
        )
        val_ratio = config.get('data.val_ratio', 0.1)
        test_ratio = config.get('data.test_ratio', 0.1)
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size,
            random_state=config.get('training.random_seed', 42)
        )

    # Create datasets
    datasets = {
        'train': ToxicityDataset(
            [graphs[i] for i in train_idx],
            [valid_smiles[i] for i in train_idx],
            [valid_scaffolds[i] for i in train_idx]
        ),
        'val': ToxicityDataset(
            [graphs[i] for i in val_idx],
            [valid_smiles[i] for i in val_idx],
            [valid_scaffolds[i] for i in val_idx]
        ),
        'test': ToxicityDataset(
            [graphs[i] for i in test_idx],
            [valid_smiles[i] for i in test_idx],
            [valid_scaffolds[i] for i in test_idx]
        )
    }

    # Create model
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

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, datasets, task_names, device, train_idx


def evaluate_model(model: torch.nn.Module,
                  dataset: ToxicityDataset,
                  task_names: List[str],
                  device: torch.device,
                  batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Evaluate model on dataset.

    Args:
        model: Trained model
        dataset: Dataset to evaluate
        task_names: List of task names
        device: Evaluation device
        batch_size: Batch size

    Returns:
        Tuple of (predictions, labels, smiles, scaffolds)
    """
    logger.info(f"Evaluating model on {len(dataset)} samples...")

    # Create data loader
    def collate_fn(batch):
        graphs, smiles, scaffolds = zip(*batch)
        from torch_geometric.data import Batch
        batch_graphs = Batch.from_data_list(graphs)
        return batch_graphs, list(smiles), list(scaffolds)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    predictions_list = []
    labels_list = []
    smiles_list = []
    scaffolds_list = []

    model.eval()
    with torch.no_grad():
        for batch_graphs, batch_smiles, batch_scaffolds in dataloader:
            batch_graphs = batch_graphs.to(device)

            # Forward pass
            outputs = model(batch_graphs)  # [batch_size, num_tasks]
            predictions = torch.sigmoid(outputs)  # Convert to probabilities

            # Store results
            predictions_list.append(predictions.cpu().numpy())
            labels_list.append(batch_graphs.y.cpu().numpy())
            smiles_list.extend(batch_smiles)
            scaffolds_list.extend(batch_scaffolds)

    # Concatenate results
    all_predictions = np.vstack(predictions_list)
    all_labels = np.vstack(labels_list)

    logger.info("Evaluation completed")

    return all_predictions, all_labels, smiles_list, scaffolds_list


def compute_comprehensive_metrics(predictions: np.ndarray,
                                labels: np.ndarray,
                                task_names: List[str],
                                smiles: List[str],
                                scaffolds: List[str],
                                train_scaffolds: Optional[List[str]] = None) -> Dict:
    """Compute comprehensive evaluation metrics.

    Args:
        predictions: Model predictions
        labels: True labels
        task_names: Task names
        smiles: SMILES strings
        scaffolds: Scaffold identifiers
        train_scaffolds: Training scaffolds for generalization analysis

    Returns:
        Dictionary of metrics
    """
    logger.info("Computing comprehensive metrics...")

    # Basic metrics
    toxicity_metrics = ToxicityMetrics(task_names=task_names)
    basic_metrics = toxicity_metrics.compute_metrics(predictions, labels)

    # Multi-task evaluation
    evaluator = MultiTaskEvaluator(task_names=task_names)
    comprehensive_metrics = evaluator.evaluate(predictions, labels)

    # Confidence intervals
    try:
        confidence_intervals = evaluator.bootstrap_confidence_intervals(
            predictions, labels, n_bootstrap=1000
        )
    except Exception as e:
        logger.warning(f"Could not compute confidence intervals: {e}")
        confidence_intervals = {}

    # Scaffold generalization analysis
    scaffold_metrics = {}
    if train_scaffolds is not None:
        logger.info("Performing scaffold generalization analysis...")
        scaffold_analyzer = ScaffoldGeneralizationAnalyzer(task_names=task_names)
        scaffold_metrics = scaffold_analyzer.analyze_scaffold_performance(
            predictions, labels, scaffolds, train_scaffolds
        )

    # Combine all metrics
    all_metrics = {
        **basic_metrics,
        **comprehensive_metrics,
        'confidence_intervals': confidence_intervals,
        'scaffold_analysis': scaffold_metrics
    }

    return all_metrics


def save_results(predictions: np.ndarray,
               labels: np.ndarray,
               smiles: List[str],
               scaffolds: List[str],
               task_names: List[str],
               metrics: Dict,
               output_dir: Path) -> None:
    """Save evaluation results to files.

    Args:
        predictions: Model predictions
        labels: True labels
        smiles: SMILES strings
        scaffolds: Scaffold identifiers
        task_names: Task names
        metrics: Computed metrics
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    results_df = pd.DataFrame({
        'smiles': smiles,
        'scaffold': scaffolds
    })

    # Add predictions and labels
    for i, task in enumerate(task_names):
        results_df[f'{task}_pred'] = predictions[:, i]
        results_df[f'{task}_true'] = labels[:, i]

    predictions_path = output_dir / 'predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save summary report
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("Molecular Toxicity Prediction - Evaluation Report\n")
        f.write("=" * 55 + "\n\n")

        f.write("Overall Performance:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean AUC-ROC: {metrics.get('auc_roc_mean', 0):.4f} ± {metrics.get('auc_roc_std', 0):.4f}\n")
        f.write(f"Mean AUC-PR:  {metrics.get('auc_pr_mean', 0):.4f} ± {metrics.get('auc_pr_std', 0):.4f}\n")
        f.write(f"Mean Accuracy: {metrics.get('accuracy_mean', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}\n")
        f.write(f"Mean F1-Score: {metrics.get('f1_mean', 0):.4f} ± {metrics.get('f1_std', 0):.4f}\n\n")

        f.write("Per-Task Performance:\n")
        f.write("-" * 22 + "\n")
        for task in task_names:
            auc_roc = metrics.get(f'auc_roc_{task}', 0)
            auc_pr = metrics.get(f'auc_pr_{task}', 0)
            f.write(f"{task:20s}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}\n")

        if 'scaffold_analysis' in metrics and metrics['scaffold_analysis']:
            f.write(f"\nScaffold Generalization:\n")
            f.write("-" * 24 + "\n")
            scaffold_gap = metrics['scaffold_analysis'].get('scaffold_split_generalization_gap', 0)
            f.write(f"Generalization Gap: {scaffold_gap:.4f}\n")
            f.write(f"Seen Scaffolds AUC: {metrics['scaffold_analysis'].get('seen_auc_roc_mean', 0):.4f}\n")
            f.write(f"Unseen Scaffolds AUC: {metrics['scaffold_analysis'].get('unseen_auc_roc_mean', 0):.4f}\n")

    logger.info(f"Saved evaluation report to {report_path}")


def create_plots(predictions: np.ndarray,
               labels: np.ndarray,
               task_names: List[str],
               scaffolds: List[str],
               train_scaffolds: Optional[List[str]],
               output_dir: Path) -> None:
    """Create evaluation plots.

    Args:
        predictions: Model predictions
        labels: True labels
        task_names: Task names
        scaffolds: Scaffold identifiers
        train_scaffolds: Training scaffolds
        output_dir: Output directory
    """
    logger.info("Creating evaluation plots...")

    # ROC curves
    toxicity_metrics = ToxicityMetrics(task_names=task_names)
    roc_fig = toxicity_metrics.plot_roc_curves(
        predictions, labels,
        save_path=output_dir / 'roc_curves.png'
    )
    plt.close(roc_fig)

    # Precision-Recall curves
    pr_fig = toxicity_metrics.plot_precision_recall_curves(
        predictions, labels,
        save_path=output_dir / 'pr_curves.png'
    )
    plt.close(pr_fig)

    # Scaffold generalization plots
    if train_scaffolds is not None:
        scaffold_analyzer = ScaffoldGeneralizationAnalyzer(task_names=task_names)
        scaffold_fig = scaffold_analyzer.plot_scaffold_performance_comparison(
            predictions, labels, scaffolds, train_scaffolds,
            save_path=output_dir / 'scaffold_generalization.png'
        )
        plt.close(scaffold_fig)

    logger.info(f"Saved plots to {output_dir}")


def main():
    """Perform comprehensive model evaluation and scaffold generalization analysis.

    Executes complete evaluation pipeline including model loading,
    prediction generation, comprehensive metrics computation,
    scaffold performance analysis, and visualization creation.

    The evaluation stages:
    1. Parse arguments and load configuration
    2. Load trained model checkpoint and test data
    3. Generate predictions on test set
    4. Compute multi-task classification metrics
    5. Analyze scaffold-based generalization performance
    6. Create performance visualizations and plots
    7. Save results and analysis outputs

    Raises:
        SystemExit: On critical errors that prevent evaluation (config, model, or data failures)
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
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

        # Override device if specified
        if args.device != 'auto':
            config['device'] = args.device

        # Set random seeds
        try:
            set_random_seeds(config.get('training.random_seed', 42))
        except Exception as e:
            logger.warning(f"Failed to set random seeds: {e}")

        # Validate checkpoint exists
        if not args.checkpoint.exists():
            logger.error(f"Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)

        # Load model and data
        try:
            model, datasets, task_names, device, train_indices = load_model_and_data(
                config, args.checkpoint, args.data_dir, args.dataset
            )
        except Exception as e:
            logger.error(f"Failed to load model and data: {e}")
            sys.exit(1)

        # Determine which split(s) to evaluate
        if args.split == 'all':
            splits_to_eval = ['train', 'val', 'test']
        else:
            splits_to_eval = [args.split]

        # Get training scaffolds for generalization analysis
        train_scaffolds = None
        if args.scaffold_analysis and 'train' in datasets:
            train_scaffolds = list(set(datasets['train'].scaffolds))

        for split in splits_to_eval:
            if split not in datasets:
                logger.warning(f"Split '{split}' not available in datasets")
                continue

            logger.info(f"Evaluating on {split} split...")

            # Evaluate model
            predictions, labels, smiles, scaffolds = evaluate_model(
                model, datasets[split], task_names, device, args.batch_size
            )

            # Compute metrics
            metrics = compute_comprehensive_metrics(
                predictions, labels, task_names, smiles, scaffolds, train_scaffolds
            )

            # Create output directory for this split
            split_output_dir = args.output_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)

            # Save results
            if args.save_predictions:
                save_results(
                    predictions, labels, smiles, scaffolds,
                    task_names, metrics, split_output_dir
                )

            # Create plots
            if args.plot_results:
                create_plots(
                    predictions, labels, task_names, scaffolds,
                    train_scaffolds, split_output_dir
                )

            # Print summary
            print(f"\n{split.upper()} Results:")
            print("=" * 30)
            print(f"Mean AUC-ROC: {metrics.get('auc_roc_mean', 0):.4f}")
            print(f"Mean AUC-PR:  {metrics.get('auc_pr_mean', 0):.4f}")
            print(f"Mean Accuracy: {metrics.get('accuracy_mean', 0):.4f}")
            print(f"Mean F1-Score: {metrics.get('f1_mean', 0):.4f}")

            if 'scaffold_analysis' in metrics and metrics['scaffold_analysis']:
                gap = metrics['scaffold_analysis'].get('scaffold_split_generalization_gap', 0)
                print(f"Scaffold Generalization Gap: {gap:.4f}")

        logger.info("Evaluation completed successfully!")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()