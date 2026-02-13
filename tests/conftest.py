"""Test fixtures and configuration for molecular toxicity prediction tests."""

import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data, DataLoader

from molecular_scaffold_aware_multi_task_toxicity_prediction.data.loader import (
    MoleculeNetLoader,
    ScaffoldSplitter,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.data.preprocessing import (
    GraphFeaturizer,
    MoleculePreprocessor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.models.model import (
    MultiTaskToxicityPredictor,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.utils.config import Config


@pytest.fixture
def sample_smiles() -> List[str]:
    """Provide sample SMILES strings for testing.

    Returns:
        List of valid SMILES strings
    """
    return [
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CC(=O)OC1=CC=CC=C1C(=O)O',        # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',    # Caffeine
        'CC1=CC=C(C=C1)C(=O)O',            # p-Toluic acid
        'C1=CC=C(C=C1)O',                  # Phenol
        'CCO',                             # Ethanol
        'CC(C)(C)O',                       # tert-Butanol
        'CCCCO',                           # 1-Butanol
    ]


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Provide sample toxicity labels for testing.

    Returns:
        Array of binary toxicity labels [n_samples, n_tasks]
    """
    # 8 samples, 3 tasks
    return np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
    ], dtype=float)


@pytest.fixture
def task_names() -> List[str]:
    """Provide sample task names.

    Returns:
        List of task names
    """
    return ['hepatotoxicity', 'cardiotoxicity', 'neurotoxicity']


@pytest.fixture
def sample_dataframe(sample_smiles: List[str], sample_labels: np.ndarray, task_names: List[str]) -> pd.DataFrame:
    """Create sample DataFrame for testing.

    Args:
        sample_smiles: SMILES strings
        sample_labels: Toxicity labels
        task_names: Task names

    Returns:
        Sample DataFrame
    """
    df = pd.DataFrame({'smiles': sample_smiles})

    for i, task in enumerate(task_names):
        df[task] = sample_labels[:, i]

    df.attrs['dataset_name'] = 'test'
    df.attrs['toxicity_columns'] = task_names
    df.attrs['n_tasks'] = len(task_names)

    return df


@pytest.fixture
def molecule_preprocessor() -> MoleculePreprocessor:
    """Provide MoleculePreprocessor instance.

    Returns:
        Configured MoleculePreprocessor
    """
    return MoleculePreprocessor(
        remove_salts=True,
        canonical_smiles=True,
        remove_stereochemistry=False
    )


@pytest.fixture
def graph_featurizer() -> GraphFeaturizer:
    """Provide GraphFeaturizer instance.

    Returns:
        Configured GraphFeaturizer
    """
    return GraphFeaturizer(
        explicit_h=False,
        use_chirality=True
    )


@pytest.fixture
def sample_graphs(sample_smiles: List[str], sample_labels: np.ndarray, graph_featurizer: GraphFeaturizer) -> List[Data]:
    """Create sample graph data objects.

    Args:
        sample_smiles: SMILES strings
        sample_labels: Toxicity labels
        graph_featurizer: Graph featurizer

    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []

    for i, smiles in enumerate(sample_smiles):
        labels = sample_labels[i]
        graph = graph_featurizer.featurize(smiles, labels)
        if graph is not None:
            graphs.append(graph)

    return graphs


@pytest.fixture
def sample_dataloader(sample_graphs: List[Data]) -> DataLoader:
    """Create sample DataLoader.

    Args:
        sample_graphs: Graph data objects

    Returns:
        PyTorch Geometric DataLoader
    """
    return DataLoader(sample_graphs, batch_size=4, shuffle=False)


@pytest.fixture
def test_config() -> Config:
    """Provide test configuration.

    Returns:
        Test configuration object
    """
    config_dict = {
        'model': {
            'name': 'gcn',
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'scaffold_dim': 32,
            'gcn': {
                'aggregation': 'add',
            },
        },
        'data': {
            'dataset': 'tox21',
            'node_dim': 133,  # Typical node feature dimension
            'batch_size': 16,
            'scaffold_split': True,
        },
        'training': {
            'num_epochs': 5,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'patience': 3,
        },
        'device': 'cpu',
    }

    return Config(config_dict)


@pytest.fixture
def temp_dir():
    """Provide temporary directory for tests.

    Yields:
        Temporary directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def scaffold_splitter() -> ScaffoldSplitter:
    """Provide ScaffoldSplitter instance.

    Returns:
        Configured ScaffoldSplitter
    """
    return ScaffoldSplitter(
        scaffold_func='murcko',
        random_state=42
    )


@pytest.fixture
def sample_model(test_config: Config, task_names: List[str]) -> MultiTaskToxicityPredictor:
    """Create sample model for testing.

    Args:
        test_config: Test configuration
        task_names: Task names

    Returns:
        Initialized model
    """
    backbone_config = {
        'node_dim': test_config['data.node_dim'],
        'hidden_dim': test_config['model.hidden_dim'],
        'num_layers': test_config['model.num_layers'],
        'dropout': test_config['model.dropout'],
        'scaffold_dim': test_config['model.scaffold_dim'],
    }

    model = MultiTaskToxicityPredictor(
        backbone='gcn',
        backbone_config=backbone_config,
        num_tasks=len(task_names),
        hidden_dims=[32, 16],
        dropout=0.2,
        use_task_embedding=True
    )

    return model


@pytest.fixture
def sample_predictions(sample_labels: np.ndarray) -> np.ndarray:
    """Generate sample predictions for testing.

    Args:
        sample_labels: True labels

    Returns:
        Sample prediction probabilities
    """
    # Add noise to true labels to simulate predictions
    noise = np.random.normal(0, 0.1, sample_labels.shape)
    predictions = np.clip(sample_labels + noise, 0, 1)

    return predictions


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


class MockDataset:
    """Mock dataset class for testing."""

    def __init__(self, data: List[Data]):
        """Initialize mock dataset.

        Args:
            data: List of graph data objects
        """
        self.data = data

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        """Get data item by index.

        Args:
            idx: Item index

        Returns:
            Graph data object
        """
        return self.data[idx]


@pytest.fixture
def mock_dataset(sample_graphs: List[Data]) -> MockDataset:
    """Create mock dataset.

    Args:
        sample_graphs: Graph data objects

    Returns:
        Mock dataset instance
    """
    return MockDataset(sample_graphs)


def create_test_yaml_config(temp_dir: Path, config_dict: Dict) -> Path:
    """Create temporary YAML config file.

    Args:
        temp_dir: Temporary directory
        config_dict: Configuration dictionary

    Returns:
        Path to created config file
    """
    import yaml

    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

    return config_path


@pytest.fixture
def test_yaml_config(temp_dir: Path, test_config: Config) -> Path:
    """Create test YAML configuration file.

    Args:
        temp_dir: Temporary directory
        test_config: Test configuration

    Returns:
        Path to YAML config file
    """
    return create_test_yaml_config(temp_dir, test_config.to_dict())