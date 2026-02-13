#!/usr/bin/env python3
"""Inference script for molecular toxicity prediction.

Loads a trained model and runs predictions on input SMILES strings.
Supports single molecule predictions or batch processing from CSV files.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molecular_scaffold_aware_multi_task_toxicity_prediction.data.loader import (
    MoleculeNetLoader,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.data.preprocessing import (
    GraphFeaturizer,
    MoleculePreprocessor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.models.model import (
    MultiTaskToxicityPredictor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.utils.config import (
    Config,
    get_device,
    get_model_config,
    load_config,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run toxicity predictions on molecular SMILES",
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
        '--smiles',
        type=str,
        help='Single SMILES string to predict'
    )

    parser.add_argument(
        '--input-file',
        type=Path,
        help='CSV file with SMILES column for batch prediction'
    )

    parser.add_argument(
        '--smiles-column',
        type=str,
        default='smiles',
        help='Name of SMILES column in input file'
    )

    parser.add_argument(
        '--output-file',
        type=Path,
        help='Path to save predictions (CSV format)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for toxicity'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for inference'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing multiple molecules'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def load_model(config: Config, checkpoint_path: Path, device: torch.device) -> tuple:
    """Load trained model from checkpoint.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, task_names)
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Get dataset info for task names
    dataset = config.get('data.dataset', required=True)
    loader = MoleculeNetLoader(data_dir=config.get('data.data_dir', './data'))
    task_info = loader.get_task_info(dataset)
    task_names = task_info['task_names']

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
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, task_names


def preprocess_smiles(smiles_list: List[str], config: Config) -> tuple:
    """Preprocess SMILES strings and create molecular graphs.

    Args:
        smiles_list: List of SMILES strings
        config: Configuration object

    Returns:
        Tuple of (graphs, valid_smiles, valid_indices)
    """
    logger.info(f"Processing {len(smiles_list)} SMILES strings...")

    preprocessor = MoleculePreprocessor(
        remove_salts=config.get('data.remove_salts', True),
        canonical_smiles=config.get('data.canonical_smiles', True)
    )

    featurizer = GraphFeaturizer(
        explicit_h=config.get('data.explicit_h', False),
        use_chirality=config.get('data.use_chirality', True)
    )

    graphs = []
    valid_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):
        # Preprocess SMILES
        processed_smiles = preprocessor.preprocess_smiles(smiles)
        if processed_smiles is None:
            logger.warning(f"Failed to process SMILES at index {idx}: {smiles}")
            continue

        # Create graph (no labels for inference)
        graph = featurizer.featurize(processed_smiles, labels=None)
        if graph is None:
            logger.warning(f"Failed to featurize SMILES at index {idx}: {smiles}")
            continue

        graphs.append(graph)
        valid_smiles.append(processed_smiles)
        valid_indices.append(idx)

    logger.info(f"Successfully processed {len(graphs)}/{len(smiles_list)} molecules")

    return graphs, valid_smiles, valid_indices


def predict_batch(model: torch.nn.Module,
                 graphs: List,
                 device: torch.device,
                 batch_size: int = 32) -> np.ndarray:
    """Run predictions on batch of molecular graphs.

    Args:
        model: Trained model
        graphs: List of molecular graphs
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        Predictions array of shape (num_molecules, num_tasks)
    """
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch

    logger.info(f"Running predictions on {len(graphs)} molecules...")

    # Create simple dataset
    class SimpleDataset:
        def __init__(self, graphs):
            self.graphs = graphs

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return self.graphs[idx]

    dataset = SimpleDataset(graphs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: Batch.from_data_list(batch)
    )

    predictions_list = []
    model.eval()

    with torch.no_grad():
        for batch_graphs in dataloader:
            batch_graphs = batch_graphs.to(device)
            outputs = model(batch_graphs)
            predictions = torch.sigmoid(outputs)
            predictions_list.append(predictions.cpu().numpy())

    all_predictions = np.vstack(predictions_list)
    logger.info("Predictions completed")

    return all_predictions


def format_predictions(predictions: np.ndarray,
                      smiles_list: List[str],
                      task_names: List[str],
                      threshold: float = 0.5) -> pd.DataFrame:
    """Format predictions as pandas DataFrame.

    Args:
        predictions: Prediction probabilities
        smiles_list: SMILES strings
        task_names: Task names
        threshold: Classification threshold

    Returns:
        DataFrame with predictions
    """
    results = {'smiles': smiles_list}

    # Add probability predictions
    for i, task in enumerate(task_names):
        results[f'{task}_prob'] = predictions[:, i]
        results[f'{task}_toxic'] = (predictions[:, i] >= threshold).astype(int)

    return pd.DataFrame(results)


def main():
    """Run toxicity prediction inference on molecular SMILES.

    Loads trained model and performs predictions on either:
    - Single SMILES string (--smiles)
    - Batch of SMILES from CSV file (--input-file)

    Outputs predictions as probabilities and binary classifications.
    """
    args = parse_args()

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate input
    if not args.smiles and not args.input_file:
        logger.error("Must provide either --smiles or --input-file")
        sys.exit(1)

    if args.smiles and args.input_file:
        logger.error("Provide only one of --smiles or --input-file")
        sys.exit(1)

    try:
        # Load configuration
        config = load_config(args.config)

        # Override device if specified
        if args.device != 'auto':
            config['device'] = args.device

        device = get_device(config)
        logger.info(f"Using device: {device}")

        # Load model
        model, task_names = load_model(config, args.checkpoint, device)

        # Get input SMILES
        if args.smiles:
            smiles_list = [args.smiles]
            logger.info(f"Processing single SMILES: {args.smiles}")
        else:
            df = pd.read_csv(args.input_file)
            if args.smiles_column not in df.columns:
                logger.error(f"Column '{args.smiles_column}' not found in {args.input_file}")
                sys.exit(1)
            smiles_list = df[args.smiles_column].tolist()
            logger.info(f"Loaded {len(smiles_list)} SMILES from {args.input_file}")

        # Preprocess molecules
        graphs, valid_smiles, valid_indices = preprocess_smiles(smiles_list, config)

        if len(graphs) == 0:
            logger.error("No valid molecules to predict")
            sys.exit(1)

        # Run predictions
        predictions = predict_batch(model, graphs, device, args.batch_size)

        # Format results
        results_df = format_predictions(
            predictions, valid_smiles, task_names, args.threshold
        )

        # Display results
        if args.smiles:
            print("\nToxicity Predictions:")
            print("=" * 60)
            for task in task_names:
                prob = results_df[f'{task}_prob'].iloc[0]
                toxic = results_df[f'{task}_toxic'].iloc[0]
                status = "TOXIC" if toxic else "NON-TOXIC"
                print(f"{task:20s}: {prob:.4f} ({status})")
        else:
            print(f"\nProcessed {len(results_df)} molecules")
            print(f"Success rate: {len(results_df)/len(smiles_list)*100:.1f}%")

        # Save results if output file specified
        if args.output_file:
            results_df.to_csv(args.output_file, index=False)
            logger.info(f"Saved predictions to {args.output_file}")
            print(f"\nPredictions saved to: {args.output_file}")

        # Print summary statistics
        if len(results_df) > 1:
            print("\nToxicity Summary:")
            print("=" * 60)
            for task in task_names:
                mean_prob = results_df[f'{task}_prob'].mean()
                toxic_count = results_df[f'{task}_toxic'].sum()
                toxic_pct = toxic_count / len(results_df) * 100
                print(f"{task:20s}: {toxic_count}/{len(results_df)} toxic ({toxic_pct:.1f}%), mean_prob={mean_prob:.4f}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
