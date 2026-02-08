"""Molecular data preprocessing and graph featurization."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

logger = logging.getLogger(__name__)


class MoleculePreprocessor:
    """Preprocessor for molecular data before graph construction.

    Handles SMILES cleaning, standardization, and basic molecular descriptors.
    """

    def __init__(self,
                 remove_salts: bool = True,
                 canonical_smiles: bool = True,
                 remove_stereochemistry: bool = False):
        """Initialize molecular preprocessor.

        Args:
            remove_salts: Whether to remove salt components
            canonical_smiles: Whether to canonicalize SMILES
            remove_stereochemistry: Whether to remove stereochemical information
        """
        self.remove_salts = remove_salts
        self.canonical_smiles = canonical_smiles
        self.remove_stereochemistry = remove_stereochemistry

    def preprocess_smiles(self, smiles: str) -> Optional[str]:
        """Preprocess a single SMILES string.

        Args:
            smiles: Input SMILES string

        Returns:
            Preprocessed SMILES string or None if invalid

        Raises:
            TypeError: If smiles is not a string
        """
        # Input validation
        if not isinstance(smiles, str):
            raise TypeError(f"Expected string, got {type(smiles)}")

        if not smiles or not smiles.strip():
            logger.debug("Empty or whitespace-only SMILES provided")
            return None

        if len(smiles) > 1000:  # Reasonable limit for SMILES length
            logger.warning(f"Unusually long SMILES ({len(smiles)} chars), may cause performance issues")

        try:
            logger.debug(f"Preprocessing SMILES: {smiles}")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.debug(f"Failed to parse SMILES: {smiles}")
                return None

            original_atoms = mol.GetNumAtoms()

            # Remove salts (keep largest fragment)
            if self.remove_salts:
                mol = self._remove_salts(mol)
                new_atoms = mol.GetNumAtoms()
                if new_atoms != original_atoms:
                    logger.debug(f"Removed salts: {original_atoms} -> {new_atoms} atoms")

            # Remove stereochemistry
            if self.remove_stereochemistry:
                Chem.RemoveStereochemistry(mol)
                logger.debug("Removed stereochemistry information")

            # Canonicalize
            if self.canonical_smiles:
                processed_smiles = Chem.MolToSmiles(mol, canonical=True)
                logger.debug(f"Canonicalized SMILES: {processed_smiles}")
            else:
                processed_smiles = Chem.MolToSmiles(mol)

            logger.debug(f"Successfully preprocessed SMILES with {mol.GetNumAtoms()} atoms")
            return processed_smiles

        except (ValueError, TypeError) as e:
            logger.warning(f"RDKit parsing error for SMILES {smiles}: {e}")
            return None
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"RDKit processing error for SMILES {smiles}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error preprocessing SMILES {smiles}: {e}")
            return None

    def _remove_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """Remove salt components, keeping the largest fragment.

        Args:
            mol: RDKit molecule

        Returns:
            Molecule with salts removed
        """
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if len(fragments) == 1:
            return mol

        # Return largest fragment by heavy atom count
        largest_fragment = max(fragments, key=lambda x: x.GetNumHeavyAtoms())
        return largest_fragment

    def compute_descriptors(self, smiles: str) -> Optional[Dict[str, float]]:
        """Compute molecular descriptors for a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of molecular descriptors
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                'formal_charge': Chem.GetFormalCharge(mol),
            }

            return descriptors

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid molecule for descriptor computation {smiles}: {e}")
        except (AttributeError, KeyError) as e:
            logger.warning(f"RDKit descriptor computation error for {smiles}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error computing descriptors for {smiles}: {e}")
            return None


class GraphFeaturizer:
    """Convert molecular SMILES to graph representations.

    Creates PyTorch Geometric Data objects with node and edge features
    suitable for graph neural networks.
    """

    # Atom feature mapping
    ATOM_FEATURES = {
        'atomic_num': list(range(1, 119)),  # 1-118
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-2, -1, 0, 1, 2],
        'chiral_tag': [0, 1, 2, 3],
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other'
        ],
        'is_aromatic': [False, True],
        'is_in_ring': [False, True],
    }

    # Bond feature mapping
    BOND_FEATURES = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOZ,
        ],
        'is_conjugated': [False, True],
        'is_in_ring': [False, True],
    }

    def __init__(self,
                 explicit_h: bool = False,
                 use_chirality: bool = True):
        """Initialize graph featurizer.

        Args:
            explicit_h: Whether to include explicit hydrogens
            use_chirality: Whether to include chirality information
        """
        self.explicit_h = explicit_h
        self.use_chirality = use_chirality

    def featurize(self,
                  smiles: str,
                  labels: Optional[Union[float, List[float]]] = None) -> Optional[Data]:
        """Convert SMILES to PyTorch Geometric Data object.

        Args:
            smiles: SMILES string
            labels: Target labels for the molecule

        Returns:
            PyTorch Geometric Data object or None if invalid

        Raises:
            TypeError: If inputs have incorrect types
            ValueError: If labels contain invalid values
        """
        # Input validation
        if not isinstance(smiles, str):
            raise TypeError(f"Expected string for smiles, got {type(smiles)}")

        if not smiles or not smiles.strip():
            logger.debug("Empty or whitespace-only SMILES provided")
            return None

        if labels is not None:
            if isinstance(labels, list):
                if not all(isinstance(x, (int, float)) for x in labels):
                    raise ValueError("All labels must be numeric")
                if any(np.isnan(x) or np.isinf(x) for x in labels if isinstance(x, (int, float))):
                    logger.warning(f"Labels contain NaN or inf values for {smiles}")
            elif isinstance(labels, (int, float)):
                if np.isnan(labels) or np.isinf(labels):
                    logger.warning(f"Label contains NaN or inf value for {smiles}")
            else:
                raise TypeError(f"Expected numeric labels, got {type(labels)}")

        try:
            logger.debug(f"Featurizing SMILES: {smiles}")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.debug(f"Failed to parse molecule from SMILES: {smiles}")
                return None

            if self.explicit_h:
                mol = Chem.AddHs(mol)
                logger.debug("Added explicit hydrogens")

            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            logger.debug(f"Molecule has {num_atoms} atoms and {num_bonds} bonds")

            # Get node features
            node_features = self._get_node_features(mol)
            if node_features is None:
                logger.debug("Failed to compute node features")
                return None

            # Get edge features and indices
            edge_indices, edge_features = self._get_edge_features(mol)

            # Compute edge feature dimension for empty graphs
            edge_feat_dim = self.get_feature_dims()['edge_dim']

            if edge_features is not None:
                edge_features_array = np.array(edge_features)
                logger.debug(f"Generated {len(edge_indices)} edges with {edge_features_array.shape[1]} features each")
            else:
                logger.debug(f"Generated {len(edge_indices)} edges with no edge features")

            # Create edge_index tensor with correct shape [2, num_edges]
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Create Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                smiles=smiles,
            )

            # Always set edge_attr for consistent batching
            if edge_features is not None:
                data.edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                data.edge_attr = torch.zeros((0, edge_feat_dim), dtype=torch.float)

            if labels is not None:
                if isinstance(labels, (int, float)):
                    data.y = torch.tensor([[labels]], dtype=torch.float)
                else:
                    data.y = torch.tensor([labels], dtype=torch.float)

            return data

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid input for featurization {smiles}: {e}")
            return None
        except (IndexError, AttributeError) as e:
            logger.warning(f"Graph construction error for {smiles}: {e}")
            return None
        except RuntimeError as e:
            logger.warning(f"PyTorch tensor error for {smiles}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected featurization error for {smiles}: {e}")
            return None

    def _get_node_features(self, mol: Chem.Mol) -> Optional[List[List[int]]]:
        """Extract node (atom) features from molecule.

        Args:
            mol: RDKit molecule

        Returns:
            List of atom feature vectors
        """
        if mol.GetNumAtoms() == 0:
            return None

        node_features = []

        for atom in mol.GetAtoms():
            features = []

            # Atomic number (one-hot)
            features.extend(self._one_hot_encode(
                atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num']))

            # Degree (one-hot)
            features.extend(self._one_hot_encode(
                atom.GetTotalDegree(), self.ATOM_FEATURES['degree']))

            # Formal charge (one-hot)
            features.extend(self._one_hot_encode(
                atom.GetFormalCharge(), self.ATOM_FEATURES['formal_charge']))

            # Hybridization (one-hot)
            hybridization = atom.GetHybridization()
            if hybridization in self.ATOM_FEATURES['hybridization']:
                hyb_value = hybridization
            else:
                hyb_value = 'other'
            features.extend(self._one_hot_encode(
                hyb_value, self.ATOM_FEATURES['hybridization']))

            # Aromaticity
            features.extend(self._one_hot_encode(
                atom.GetIsAromatic(), self.ATOM_FEATURES['is_aromatic']))

            # In ring
            features.extend(self._one_hot_encode(
                atom.IsInRing(), self.ATOM_FEATURES['is_in_ring']))

            # Chirality (if enabled)
            if self.use_chirality:
                features.extend(self._one_hot_encode(
                    int(atom.GetChiralTag()), self.ATOM_FEATURES['chiral_tag']))

            # Additional features
            features.append(atom.GetMass())
            features.append(atom.GetTotalValence())
            features.append(atom.GetNumRadicalElectrons())

            node_features.append(features)

        return node_features

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        """Extract edge (bond) features from molecule.

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (edge_indices, edge_features)
        """
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()

            # Add both directions for undirected graph
            edge_indices.extend([[start_atom, end_atom], [end_atom, start_atom]])

            # Bond features
            features = []

            # Bond type (one-hot)
            features.extend(self._one_hot_encode(
                bond.GetBondType(), self.BOND_FEATURES['bond_type']))

            # Stereo (one-hot)
            features.extend(self._one_hot_encode(
                bond.GetStereo(), self.BOND_FEATURES['stereo']))

            # Conjugated
            features.extend(self._one_hot_encode(
                bond.GetIsConjugated(), self.BOND_FEATURES['is_conjugated']))

            # In ring
            features.extend(self._one_hot_encode(
                bond.IsInRing(), self.BOND_FEATURES['is_in_ring']))

            # Add same features for both directions
            edge_features.extend([features, features])

        if not edge_features:
            return edge_indices, None

        return edge_indices, edge_features

    def _one_hot_encode(self, value, allowed_values: List) -> List[int]:
        """Create one-hot encoding for a value.

        Args:
            value: Value to encode
            allowed_values: List of allowed values

        Returns:
            One-hot encoded vector
        """
        encoding = [0] * (len(allowed_values) + 1)  # +1 for unknown

        try:
            index = allowed_values.index(value)
            encoding[index] = 1
        except ValueError:
            # Unknown value
            encoding[-1] = 1

        return encoding

    def get_feature_dims(self) -> Dict[str, int]:
        """Get dimensions of node and edge features.

        Returns:
            Dictionary with feature dimensions
        """
        # Calculate node feature dimension
        node_dim = 0
        node_dim += len(self.ATOM_FEATURES['atomic_num']) + 1  # +1 for unknown
        node_dim += len(self.ATOM_FEATURES['degree']) + 1
        node_dim += len(self.ATOM_FEATURES['formal_charge']) + 1
        node_dim += len(self.ATOM_FEATURES['hybridization']) + 1
        node_dim += len(self.ATOM_FEATURES['is_aromatic']) + 1
        node_dim += len(self.ATOM_FEATURES['is_in_ring']) + 1

        if self.use_chirality:
            node_dim += len(self.ATOM_FEATURES['chiral_tag']) + 1

        node_dim += 3  # mass, valence, radical electrons

        # Calculate edge feature dimension
        edge_dim = 0
        edge_dim += len(self.BOND_FEATURES['bond_type']) + 1
        edge_dim += len(self.BOND_FEATURES['stereo']) + 1
        edge_dim += len(self.BOND_FEATURES['is_conjugated']) + 1
        edge_dim += len(self.BOND_FEATURES['is_in_ring']) + 1

        return {
            'node_dim': node_dim,
            'edge_dim': edge_dim
        }


class ScaffoldAwareTransform(BaseTransform):
    """Transform that adds scaffold information to molecular graphs.

    This transform computes scaffold embeddings and adds them as additional
    node or graph-level features for scaffold-aware models.
    """

    def __init__(self,
                 scaffold_type: str = 'murcko',
                 embedding_dim: int = 64):
        """Initialize scaffold-aware transform.

        Args:
            scaffold_type: Type of scaffold to compute
            embedding_dim: Dimension of scaffold embeddings
        """
        self.scaffold_type = scaffold_type
        self.embedding_dim = embedding_dim
        self._scaffold_embeddings = {}

    def __call__(self, data: Data) -> Data:
        """Apply scaffold-aware transform to data.

        Args:
            data: Input graph data

        Returns:
            Data with scaffold features added
        """
        if not hasattr(data, 'smiles'):
            return data

        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                return data

            # Compute scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)

            # Get or create scaffold embedding
            if scaffold_smiles not in self._scaffold_embeddings:
                # Simple hash-based embedding (in practice, could use learned embeddings)
                scaffold_hash = hash(scaffold_smiles) % (2**32)
                rng = np.random.RandomState(scaffold_hash)
                embedding = torch.tensor(
                    rng.randn(1, self.embedding_dim),
                    dtype=torch.float
                )
                self._scaffold_embeddings[scaffold_smiles] = embedding

            scaffold_embedding = self._scaffold_embeddings[scaffold_smiles]

            # Add scaffold embedding as graph-level feature
            data.scaffold_embedding = scaffold_embedding
            data.scaffold_smiles = scaffold_smiles

        except (ValueError, TypeError) as e:
            # Catch RDKit parsing/computation errors
            logger.warning(f"Failed to compute scaffold for {data.smiles} due to molecule parsing error: {e}")
        except (AttributeError, KeyError) as e:
            # Catch data structure issues
            logger.warning(f"Failed to compute scaffold for {data.smiles} due to data structure error: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            logger.warning(f"Unexpected error computing scaffold for {data.smiles}: {e}")

        # Ensure scaffold_embedding is always set for consistent batching
        if not hasattr(data, 'scaffold_embedding') or data.scaffold_embedding is None:
            data.scaffold_embedding = torch.zeros(1, self.embedding_dim, dtype=torch.float)
            data.scaffold_smiles = ""

        return data

    def forward(self, data: Data) -> Data:
        """Forward method required by BaseTransform.

        Args:
            data: Input graph data

        Returns:
            Data with scaffold features added
        """
        return self.__call__(data)