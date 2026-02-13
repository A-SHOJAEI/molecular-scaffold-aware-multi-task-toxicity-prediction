"""Data loading utilities for molecular toxicity datasets."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import TUDataset

logger = logging.getLogger(__name__)


class MoleculeNetLoader:
    """Loader for MoleculeNet toxicity datasets (Tox21, ToxCast, ClinTox).

    This class provides unified interface for loading and processing different
    toxicity datasets from MoleculeNet benchmark.
    """

    DATASET_URLS = {
        'tox21': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
        'toxcast': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz',
        'clintox': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
    }

    def __init__(self,
                 data_dir: Union[str, Path] = "./data",
                 cache: bool = True,
                 timeout: int = 60,
                 max_retries: int = 3):
        """Initialize the MoleculeNet loader.

        Args:
            data_dir: Directory to store downloaded datasets
            cache: Whether to cache processed datasets
            timeout: Network timeout in seconds for downloads
            max_retries: Maximum number of retry attempts for failed downloads
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        self.timeout = timeout
        self.max_retries = max_retries
        self._datasets = {}

    def load_dataset(self,
                    name: str,
                    force_reload: bool = False) -> pd.DataFrame:
        """Load a specific toxicity dataset.

        Args:
            name: Dataset name ('tox21', 'toxcast', 'clintox')
            force_reload: Whether to force reload from source

        Returns:
            DataFrame with molecular data and toxicity labels

        Raises:
            ValueError: If dataset name is not supported
        """
        if name not in self.DATASET_URLS:
            raise ValueError(f"Dataset {name} not supported. "
                           f"Available: {list(self.DATASET_URLS.keys())}")

        cache_path = self.data_dir / f"{name}_processed.pkl"

        if not force_reload and self.cache and cache_path.exists():
            logger.info(f"Loading cached dataset: {name}")
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            self._datasets[name] = df
            return df

        logger.info(f"Loading dataset from source: {name}")
        logger.debug(f"Downloading from URL: {self.DATASET_URLS[name]}")

        # Add timeout and retry logic for network downloads
        try:
            import time
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            download_start = time.time()

            session = requests.Session()
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            response = session.get(self.DATASET_URLS[name], timeout=self.timeout)
            response.raise_for_status()

            download_time = time.time() - download_start
            content_size = len(response.content) / (1024 * 1024)  # MB
            logger.info(f"Downloaded {name} dataset: {content_size:.2f} MB in {download_time:.2f}s")

            from io import BytesIO
            url = self.DATASET_URLS[name]
            compression = 'gzip' if url.endswith('.gz') else None
            df = pd.read_csv(BytesIO(response.content), compression=compression)

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            logger.error(f"Failed to download dataset {name} from {self.DATASET_URLS[name]}: {e}")
            raise RuntimeError(f"Network error downloading dataset {name}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error downloading dataset {name}: {e}")
            raise RuntimeError(f"Failed to download dataset {name}: {e}") from e
        logger.info(f"Downloaded dataset with {len(df)} rows and {len(df.columns)} columns")

        # Clean and preprocess data
        logger.info(f"Preprocessing {name} dataset...")
        df = self._preprocess_dataset(df, name)

        if self.cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Cached preprocessed dataset to {cache_path}")

        self._datasets[name] = df
        return df

    def _preprocess_dataset(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Preprocess dataset-specific formatting.

        Args:
            df: Raw dataset DataFrame
            name: Dataset name

        Returns:
            Preprocessed DataFrame with consistent columns
        """
        # Remove invalid SMILES
        initial_count = len(df)
        df = df.dropna(subset=['smiles'])
        logger.debug(f"Removed {initial_count - len(df)} rows with missing SMILES")

        # Validate SMILES
        valid_mask = df['smiles'].apply(self._is_valid_smiles)
        valid_count = valid_mask.sum()
        df = df[valid_mask].reset_index(drop=True)
        logger.info(f"Kept {valid_count}/{len(valid_mask)} molecules with valid SMILES")

        # Dataset-specific preprocessing
        toxicity_cols = []  # Initialize to avoid UnboundLocalError

        if name == 'tox21':
            # Tox21 has 12 toxicity endpoints (NR- and SR- prefixed)
            toxicity_cols = [col for col in df.columns if col.startswith('NR-') or col.startswith('SR-')]
            df[toxicity_cols] = df[toxicity_cols].fillna(0)  # Missing as non-toxic

        elif name == 'toxcast':
            # ToxCast has many assays - select high-quality ones
            assay_cols = [col for col in df.columns if 'assay' in col.lower()]
            if len(assay_cols) > 50:  # Limit to manageable number
                assay_cols = assay_cols[:50]
            toxicity_cols = assay_cols
            df[toxicity_cols] = df[toxicity_cols].fillna(0)

        elif name == 'clintox':
            # ClinTox has FDA approval and toxicity labels
            toxicity_cols = ['FDA_APPROVED', 'CT_TOX']
            df[toxicity_cols] = df[toxicity_cols].fillna(0)

        else:
            # For unknown datasets, try to infer toxicity columns
            toxicity_cols = [col for col in df.columns if col != 'smiles']
            logger.warning(f"Unknown dataset {name}, using all non-SMILES columns as toxicity targets: {toxicity_cols}")
            df[toxicity_cols] = df[toxicity_cols].fillna(0)

        # Store metadata
        df.attrs['dataset_name'] = name
        df.attrs['toxicity_columns'] = toxicity_cols
        df.attrs['n_tasks'] = len(toxicity_cols)

        logger.info(f"Preprocessed {name}: {len(df)} molecules, "
                   f"{len(toxicity_cols)} toxicity tasks")

        return df

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES string is valid.

        Args:
            smiles: SMILES string

        Returns:
            True if valid SMILES
        """
        if not smiles or not smiles.strip():
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() > 0
        except (ValueError, TypeError) as e:
            # Catch RDKit parsing errors for invalid SMILES
            logger.debug(f"Invalid SMILES '{smiles}': {e}")
            return False
        except (AttributeError, KeyError, RuntimeError) as e:
            # Catch RDKit internal errors and attribute access issues
            logger.debug(f"RDKit error processing SMILES '{smiles}': {e}")
            return False
        except Exception as e:
            # Catch any other unexpected errors as last resort
            logger.warning(f"Unexpected error validating SMILES '{smiles}': {e}")
            return False

    def get_task_info(self, name: str) -> Dict[str, Any]:
        """Get information about tasks in a dataset.

        Args:
            name: Dataset name

        Returns:
            Dictionary with task information
        """
        if name not in self._datasets:
            self.load_dataset(name)

        df = self._datasets[name]
        toxicity_cols = df.attrs['toxicity_columns']

        task_info = {
            'n_tasks': len(toxicity_cols),
            'task_names': toxicity_cols,
            'task_types': ['binary_classification'] * len(toxicity_cols),
            'class_distribution': {}
        }

        for col in toxicity_cols:
            task_info['class_distribution'][col] = df[col].value_counts().to_dict()

        return task_info


class ScaffoldSplitter:
    """Scaffold-aware molecular data splitter.

    This class implements scaffold-based splitting to ensure that molecules
    with similar scaffolds are not split across train/validation/test sets,
    providing a more realistic evaluation of model generalization.
    """

    def __init__(self,
                 scaffold_func: str = 'murcko',
                 random_state: int = 42):
        """Initialize scaffold splitter.

        Args:
            scaffold_func: Scaffold function to use ('murcko', 'brics')
            random_state: Random state for reproducible splits
        """
        self.scaffold_func = scaffold_func
        self.random_state = random_state
        np.random.seed(random_state)

    def split(self,
              smiles: List[str],
              train_ratio: float = 0.8,
              val_ratio: float = 0.1,
              test_ratio: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
        """Split molecules based on molecular scaffolds.

        Args:
            smiles: List of SMILES strings
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        # Generate scaffolds for all molecules
        scaffolds = self._generate_scaffolds(smiles)

        # Group molecules by scaffold
        scaffold_to_indices = {}
        for idx, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = []
            scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds by size (largest first) for balanced splits
        scaffold_sets = [(scaffold, indices) for scaffold, indices
                        in scaffold_to_indices.items()]
        scaffold_sets.sort(key=lambda x: len(x[1]), reverse=True)

        # Greedily assign scaffolds to splits
        train_indices, val_indices, test_indices = [], [], []
        train_size, val_size, test_size = 0, 0, 0
        total_size = len(smiles)

        for scaffold, indices in scaffold_sets:
            current_train_ratio = train_size / total_size if total_size > 0 else 0
            current_val_ratio = val_size / total_size if total_size > 0 else 0
            current_test_ratio = test_size / total_size if total_size > 0 else 0

            # Assign to the split that needs more data
            if current_train_ratio < train_ratio:
                train_indices.extend(indices)
                train_size += len(indices)
            elif current_val_ratio < val_ratio:
                val_indices.extend(indices)
                val_size += len(indices)
            else:
                test_indices.extend(indices)
                test_size += len(indices)

        logger.info(f"Scaffold split: train={len(train_indices)}, "
                   f"val={len(val_indices)}, test={len(test_indices)}")
        logger.info(f"Unique scaffolds: train={len(set(scaffolds[i] for i in train_indices))}, "
                   f"val={len(set(scaffolds[i] for i in val_indices))}, "
                   f"test={len(set(scaffolds[i] for i in test_indices))}")

        return train_indices, val_indices, test_indices

    def _generate_scaffolds(self, smiles: List[str]) -> List[str]:
        """Generate molecular scaffolds.

        Args:
            smiles: List of SMILES strings

        Returns:
            List of scaffold SMILES
        """
        scaffolds = []

        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    scaffolds.append("")
                    continue

                if self.scaffold_func == 'murcko':
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                else:
                    raise ValueError(f"Unknown scaffold function: {self.scaffold_func}")

                scaffolds.append(scaffold_smiles)

            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Failed to generate scaffold for {smi}: {e}")
                scaffolds.append("")
            except RuntimeError as e:
                logger.error(f"RDKit runtime error for scaffold generation on {smi}: {e}")
                scaffolds.append("")
            except Exception as e:
                logger.error(f"Unexpected error generating scaffold for {smi}: {e}")
                scaffolds.append("")

        return scaffolds

    def analyze_scaffold_diversity(self, smiles: List[str]) -> Dict[str, Any]:
        """Analyze scaffold diversity in the dataset.

        Args:
            smiles: List of SMILES strings

        Returns:
            Dictionary with diversity statistics
        """
        scaffolds = self._generate_scaffolds(smiles)
        unique_scaffolds = set(scaffold for scaffold in scaffolds if scaffold)

        scaffold_counts = {}
        for scaffold in scaffolds:
            if scaffold:
                scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1

        stats = {
            'total_molecules': len(smiles),
            'unique_scaffolds': len(unique_scaffolds),
            'scaffold_ratio': len(unique_scaffolds) / len(smiles),
            'avg_molecules_per_scaffold': len(smiles) / len(unique_scaffolds),
            'largest_scaffold_size': max(scaffold_counts.values()) if scaffold_counts else 0,
            'singleton_scaffolds': sum(1 for count in scaffold_counts.values() if count == 1)
        }

        return stats