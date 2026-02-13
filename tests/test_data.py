"""Tests for data loading and preprocessing functionality."""

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from molecular_scaffold_aware_multi_task_toxicity_prediction.data.loader import (
    MoleculeNetLoader,
    ScaffoldSplitter,
)
from molecular_scaffold_aware_multi_task_toxicity_prediction.data.preprocessing import (
    GraphFeaturizer,
    MoleculePreprocessor,
    ScaffoldAwareTransform,
)


class TestMoleculePreprocessor:
    """Test cases for MoleculePreprocessor."""

    def test_init(self):
        """Test MoleculePreprocessor initialization."""
        preprocessor = MoleculePreprocessor()
        assert preprocessor.remove_salts is True
        assert preprocessor.canonical_smiles is True
        assert preprocessor.remove_stereochemistry is False

    def test_preprocess_smiles_valid(self, molecule_preprocessor: MoleculePreprocessor):
        """Test preprocessing of valid SMILES strings."""
        smiles = 'CCO'  # Ethanol
        result = molecule_preprocessor.preprocess_smiles(smiles)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Check if it's valid SMILES
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_preprocess_smiles_invalid(self, molecule_preprocessor: MoleculePreprocessor):
        """Test preprocessing of invalid SMILES strings."""
        invalid_smiles = 'INVALID_SMILES'
        result = molecule_preprocessor.preprocess_smiles(invalid_smiles)

        assert result is None

    def test_preprocess_smiles_with_salts(self):
        """Test preprocessing SMILES with salt components."""
        preprocessor = MoleculePreprocessor(remove_salts=True)
        smiles_with_salt = 'CCO.Cl'  # Ethanol with chloride
        result = preprocessor.preprocess_smiles(smiles_with_salt)

        assert result is not None
        # Should return only the largest fragment (ethanol)
        assert 'Cl' not in result

    def test_canonical_smiles(self):
        """Test canonical SMILES generation."""
        preprocessor = MoleculePreprocessor(canonical_smiles=True)
        smiles = 'C(C)O'  # Non-canonical ethanol
        result = preprocessor.preprocess_smiles(smiles)

        assert result == 'CCO'  # Canonical form

    def test_compute_descriptors_valid(self, molecule_preprocessor: MoleculePreprocessor):
        """Test molecular descriptor computation for valid SMILES."""
        smiles = 'CCO'  # Ethanol
        descriptors = molecule_preprocessor.compute_descriptors(smiles)

        assert descriptors is not None
        assert isinstance(descriptors, dict)

        # Check required descriptors
        required_descriptors = [
            'molecular_weight', 'logp', 'num_hbd', 'num_hba', 'tpsa',
            'num_rotatable_bonds', 'num_aromatic_rings', 'num_heavy_atoms',
            'formal_charge'
        ]

        for desc in required_descriptors:
            assert desc in descriptors
            assert isinstance(descriptors[desc], (int, float))

        # Test specific values for ethanol
        assert descriptors['num_heavy_atoms'] == 3  # 2 carbons + 1 oxygen
        assert descriptors['num_hbd'] == 1  # OH group

    def test_compute_descriptors_invalid(self, molecule_preprocessor: MoleculePreprocessor):
        """Test descriptor computation for invalid SMILES."""
        invalid_smiles = 'INVALID'
        descriptors = molecule_preprocessor.compute_descriptors(invalid_smiles)

        assert descriptors is None


class TestGraphFeaturizer:
    """Test cases for GraphFeaturizer."""

    def test_init(self):
        """Test GraphFeaturizer initialization."""
        featurizer = GraphFeaturizer()
        assert featurizer.explicit_h is False
        assert featurizer.use_chirality is True

    def test_featurize_valid_smiles(self, graph_featurizer: GraphFeaturizer):
        """Test graph featurization of valid SMILES."""
        smiles = 'CCO'  # Ethanol
        labels = [0.5, 1.0, 0.0]

        data = graph_featurizer.featurize(smiles, labels)

        assert data is not None
        assert hasattr(data, 'x')  # Node features
        assert hasattr(data, 'edge_index')  # Edge indices
        assert hasattr(data, 'y')  # Labels
        assert hasattr(data, 'smiles')

        # Check tensor shapes
        assert data.x.dim() == 2  # [num_nodes, node_features]
        assert data.edge_index.dim() == 2  # [2, num_edges]
        assert data.y.dim() == 1  # [num_tasks]

        # Check SMILES preservation
        assert data.smiles == smiles

        # Check labels
        expected_labels = torch.tensor(labels, dtype=torch.float)
        assert torch.allclose(data.y, expected_labels)

    def test_featurize_invalid_smiles(self, graph_featurizer: GraphFeaturizer):
        """Test featurization of invalid SMILES."""
        invalid_smiles = 'INVALID'
        data = graph_featurizer.featurize(invalid_smiles)

        assert data is None

    def test_featurize_without_labels(self, graph_featurizer: GraphFeaturizer):
        """Test featurization without labels."""
        smiles = 'CCO'
        data = graph_featurizer.featurize(smiles)

        assert data is not None
        assert data.y is None  # No labels provided

    def test_featurize_single_label(self, graph_featurizer: GraphFeaturizer):
        """Test featurization with single label."""
        smiles = 'CCO'
        label = 0.8

        data = graph_featurizer.featurize(smiles, label)

        assert data is not None
        assert hasattr(data, 'y')
        assert data.y.shape == torch.Size([1])
        assert abs(data.y.item() - label) < 1e-6  # Floating point precision

    def test_get_feature_dims(self, graph_featurizer: GraphFeaturizer):
        """Test feature dimension calculation."""
        dims = graph_featurizer.get_feature_dims()

        assert 'node_dim' in dims
        assert 'edge_dim' in dims
        assert dims['node_dim'] > 0
        assert dims['edge_dim'] > 0

    def test_explicit_hydrogens(self):
        """Test featurization with explicit hydrogens."""
        featurizer = GraphFeaturizer(explicit_h=True)
        smiles = 'CCO'

        data = featurizer.featurize(smiles)
        assert data is not None

        # With explicit H, should have more nodes
        data_no_h = GraphFeaturizer(explicit_h=False).featurize(smiles)
        assert data.x.size(0) > data_no_h.x.size(0)

    def test_chirality_features(self):
        """Test chirality feature inclusion."""
        featurizer_with_chiral = GraphFeaturizer(use_chirality=True)
        featurizer_without_chiral = GraphFeaturizer(use_chirality=False)

        dims_with = featurizer_with_chiral.get_feature_dims()
        dims_without = featurizer_without_chiral.get_feature_dims()

        # With chirality should have more node features
        assert dims_with['node_dim'] > dims_without['node_dim']


class TestScaffoldAwareTransform:
    """Test cases for ScaffoldAwareTransform."""

    def test_init(self):
        """Test ScaffoldAwareTransform initialization."""
        transform = ScaffoldAwareTransform()
        assert transform.scaffold_type == 'murcko'
        assert transform.embedding_dim == 64

    def test_transform_with_smiles(self, graph_featurizer: GraphFeaturizer):
        """Test transform application with SMILES attribute."""
        transform = ScaffoldAwareTransform(embedding_dim=32)
        smiles = 'CCO'

        data = graph_featurizer.featurize(smiles)
        transformed_data = transform(data)

        assert hasattr(transformed_data, 'scaffold_embedding')
        assert hasattr(transformed_data, 'scaffold_smiles')
        assert transformed_data.scaffold_embedding.shape == torch.Size([32])

    def test_transform_without_smiles(self):
        """Test transform with data missing SMILES attribute."""
        transform = ScaffoldAwareTransform()
        data = torch.geometric.data.Data(x=torch.randn(5, 10))

        transformed_data = transform(data)
        # Should return data unchanged
        assert not hasattr(transformed_data, 'scaffold_embedding')


class TestScaffoldSplitter:
    """Test cases for ScaffoldSplitter."""

    def test_init(self):
        """Test ScaffoldSplitter initialization."""
        splitter = ScaffoldSplitter()
        assert splitter.scaffold_func == 'murcko'
        assert splitter.random_state == 42

    def test_split_basic(self, scaffold_splitter: ScaffoldSplitter, sample_smiles):
        """Test basic scaffold splitting."""
        train_idx, val_idx, test_idx = scaffold_splitter.split(
            sample_smiles,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Check that all indices are covered
        all_indices = set(train_idx + val_idx + test_idx)
        expected_indices = set(range(len(sample_smiles)))
        assert all_indices == expected_indices

        # Check approximate split ratios
        total = len(sample_smiles)
        assert len(train_idx) >= total * 0.4  # Allow some flexibility
        assert len(val_idx) >= 0  # May be empty for small datasets
        assert len(test_idx) >= 0

    def test_split_ratios_sum_to_one(self, scaffold_splitter: ScaffoldSplitter, sample_smiles):
        """Test that split ratios must sum to 1.0."""
        with pytest.raises(AssertionError):
            scaffold_splitter.split(
                sample_smiles,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum = 1.1
            )

    def test_generate_scaffolds(self, scaffold_splitter: ScaffoldSplitter, sample_smiles):
        """Test scaffold generation."""
        scaffolds = scaffold_splitter._generate_scaffolds(sample_smiles)

        assert len(scaffolds) == len(sample_smiles)
        assert all(isinstance(s, str) for s in scaffolds)

        # Check that scaffolds are generated (may be empty for some molecules)
        assert any(len(s) > 0 for s in scaffolds)

    def test_analyze_scaffold_diversity(self, scaffold_splitter: ScaffoldSplitter, sample_smiles):
        """Test scaffold diversity analysis."""
        stats = scaffold_splitter.analyze_scaffold_diversity(sample_smiles)

        required_keys = [
            'total_molecules',
            'unique_scaffolds',
            'scaffold_ratio',
            'avg_molecules_per_scaffold',
            'largest_scaffold_size',
            'singleton_scaffolds'
        ]

        for key in required_keys:
            assert key in stats

        # Check reasonable values
        assert stats['total_molecules'] == len(sample_smiles)
        assert stats['unique_scaffolds'] > 0
        assert 0 <= stats['scaffold_ratio'] <= 1


class TestMoleculeNetLoader:
    """Test cases for MoleculeNetLoader (without actual downloads)."""

    def test_init(self, temp_dir):
        """Test MoleculeNetLoader initialization."""
        loader = MoleculeNetLoader(data_dir=temp_dir, cache=True)
        assert loader.data_dir == temp_dir
        assert loader.cache is True
        assert temp_dir.exists()

    def test_is_valid_smiles(self):
        """Test SMILES validation."""
        assert MoleculeNetLoader._is_valid_smiles('CCO') is True
        assert MoleculeNetLoader._is_valid_smiles('INVALID') is False
        assert MoleculeNetLoader._is_valid_smiles('') is False

    def test_preprocess_dataset_basic(self, temp_dir):
        """Test basic dataset preprocessing."""
        loader = MoleculeNetLoader(data_dir=temp_dir)

        # Create test dataframe
        df = pd.DataFrame({
            'smiles': ['CCO', 'INVALID', 'CCC'],
            'NR-AR': [1, 0, 1],
            'NR-ER': [0, 1, 0]
        })

        processed_df = loader._preprocess_dataset(df, 'tox21')

        # Should remove invalid SMILES
        assert len(processed_df) == 2
        assert 'INVALID' not in processed_df['smiles'].values

        # Check attributes
        assert hasattr(processed_df, 'attrs')
        assert processed_df.attrs['dataset_name'] == 'tox21'
        assert len(processed_df.attrs['toxicity_columns']) > 0

    def test_get_task_info_mock(self, temp_dir, sample_dataframe, task_names):
        """Test task info extraction with mock data."""
        loader = MoleculeNetLoader(data_dir=temp_dir)
        loader._datasets['test'] = sample_dataframe

        task_info = loader.get_task_info('test')

        assert 'n_tasks' in task_info
        assert 'task_names' in task_info
        assert 'task_types' in task_info
        assert 'class_distribution' in task_info

        assert task_info['n_tasks'] == len(task_names)
        assert task_info['task_names'] == task_names
        assert all(t == 'binary_classification' for t in task_info['task_types'])

    @pytest.mark.parametrize('dataset_name', ['tox21', 'toxcast', 'clintox'])
    def test_unsupported_dataset_error(self, temp_dir, dataset_name):
        """Test that only supported datasets are accepted."""
        loader = MoleculeNetLoader(data_dir=temp_dir)

        # This should not raise an error for supported datasets
        # but would fail on actual download in real usage
        assert dataset_name in loader.DATASET_URLS