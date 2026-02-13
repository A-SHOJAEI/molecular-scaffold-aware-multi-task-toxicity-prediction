"""Tests for training and evaluation scripts."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import script modules
import scripts.train as train_script
import scripts.evaluate as evaluate_script


class TestTrainScript:
    """Test cases for training script functions."""

    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        with patch('sys.argv', ['train.py', '--config', '/path/to/config.yaml']):
            args = train_script.parse_args()
            assert args.config == Path('/path/to/config.yaml')
            assert args.data_dir == Path('./data')
            assert args.output_dir == Path('./outputs')

    def test_parse_args_all_options(self):
        """Test argument parsing with all options."""
        test_args = [
            'train.py',
            '--config', '/path/to/config.yaml',
            '--data-dir', '/custom/data',
            '--output-dir', '/custom/output',
            '--experiment-name', 'test_experiment',
            '--run-name', 'test_run',
            '--device', 'cpu',
            '--num-workers', '2',
            '--debug',
            '--eval-only'
        ]

        with patch('sys.argv', test_args):
            args = train_script.parse_args()
            assert args.config == Path('/path/to/config.yaml')
            assert args.data_dir == Path('/custom/data')
            assert args.output_dir == Path('/custom/output')
            assert args.experiment_name == 'test_experiment'
            assert args.run_name == 'test_run'
            assert args.device == 'cpu'
            assert args.num_workers == 2
            assert args.debug is True
            assert args.eval_only is True

    def test_toxicity_dataset(self):
        """Test ToxicityDataset class."""
        # Mock graph data
        mock_graphs = [MagicMock() for _ in range(3)]

        dataset = train_script.ToxicityDataset(mock_graphs)

        assert len(dataset) == 3
        assert dataset[0] == mock_graphs[0]
        assert dataset[2] == mock_graphs[2]

    @patch('scripts.train.MoleculeNetLoader')
    @patch('scripts.train.MoleculePreprocessor')
    @patch('scripts.train.GraphFeaturizer')
    @patch('scripts.train.ScaffoldSplitter')
    def test_load_and_preprocess_data_success(self, mock_splitter, mock_featurizer,
                                             mock_preprocessor, mock_loader):
        """Test successful data loading and preprocessing."""
        # Mock config
        config = MagicMock()
        config.get.side_effect = lambda key, default=None, required=False: {
            'data.dataset': 'tox21',
            'data.remove_salts': True,
            'data.canonical_smiles': True,
            'data.remove_stereochemistry': False,
            'data.explicit_h': False,
            'data.use_chirality': True,
            'data.use_scaffold_transform': False,
            'data.scaffold_split': True,
            'data.scaffold_func': 'murcko',
            'training.random_seed': 42,
            'data.train_ratio': 0.8,
            'data.val_ratio': 0.1,
            'data.test_ratio': 0.1
        }.get(key, default)

        # Mock loader
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance

        # Mock DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CCC'],
            'task1': [1, 0, 1],
            'task2': [0, 1, 0]
        })
        mock_loader_instance.load_dataset.return_value = mock_df
        mock_loader_instance.get_task_info.return_value = {
            'task_names': ['task1', 'task2']
        }

        # Mock preprocessor
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_preprocessor_instance.preprocess_smiles.side_effect = lambda x: x

        # Mock featurizer
        mock_featurizer_instance = MagicMock()
        mock_featurizer.return_value = mock_featurizer_instance
        mock_graph = MagicMock()
        mock_featurizer_instance.featurize.return_value = mock_graph

        # Mock splitter
        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance
        mock_splitter_instance.split.return_value = ([0], [1], [2])
        mock_splitter_instance.analyze_scaffold_diversity.return_value = {
            'unique_scaffolds': 3,
            'scaffold_ratio': 1.0
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / 'data'
            datasets, task_names = train_script.load_and_preprocess_data(config, data_dir)

            assert 'train' in datasets
            assert 'val' in datasets
            assert 'test' in datasets
            assert task_names == ['task1', 'task2']
            assert len(datasets['train']) == 1
            assert len(datasets['val']) == 1
            assert len(datasets['test']) == 1

    def test_load_and_preprocess_data_invalid_dataset(self):
        """Test data loading with invalid dataset configuration."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None, required=False: {
            'data.dataset': ''
        }.get(key, default)

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            with pytest.raises(ValueError, match="Dataset name not specified"):
                train_script.load_and_preprocess_data(config, data_dir)

    @patch('scripts.train.DataLoader')
    def test_create_data_loaders(self, mock_dataloader):
        """Test data loader creation."""
        # Mock datasets
        datasets = {
            'train': MagicMock(),
            'val': MagicMock(),
            'test': MagicMock()
        }

        # Mock config
        config = MagicMock()
        config.get.return_value = 32  # batch_size

        # Mock DataLoader
        mock_dataloader.return_value = MagicMock()

        loaders = train_script.create_data_loaders(datasets, config, num_workers=4)

        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
        assert mock_dataloader.call_count == 3

    @patch('scripts.train.MultiTaskToxicityPredictor')
    @patch('scripts.train.get_model_config')
    def test_create_model(self, mock_get_config, mock_model_class):
        """Test model creation."""
        # Mock config
        config = MagicMock()
        config.get.side_effect = lambda key, default=None, required=False: {
            'model.name': 'gcn',
            'model.prediction_hidden_dims': [128, 64],
            'model.prediction_dropout': 0.3,
            'model.use_task_embedding': True
        }.get(key, default)

        mock_get_config.return_value = {'hidden_dim': 128}
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        import torch
        device = torch.device('cpu')
        task_names = ['task1', 'task2']

        model = train_script.create_model(config, task_names, device)

        assert model == mock_model
        mock_model.to.assert_called_once_with(device)

    def test_create_optimizer_and_scheduler_adam(self):
        """Test optimizer and scheduler creation with Adam."""
        import torch.nn as nn
        from torch.optim import Adam

        model = nn.Linear(10, 1)
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            'training.optimizer': 'adam',
            'training.learning_rate': 0.001,
            'training.weight_decay': 1e-5,
            'training.scheduler': None
        }.get(key, default)

        optimizer, scheduler = train_script.create_optimizer_and_scheduler(model, config)

        assert isinstance(optimizer, Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert scheduler is None

    def test_create_optimizer_and_scheduler_adamw_with_plateau(self):
        """Test optimizer and scheduler creation with AdamW and ReduceLROnPlateau."""
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        model = nn.Linear(10, 1)
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            'training.optimizer': 'adamw',
            'training.learning_rate': 0.001,
            'training.weight_decay': 1e-4,
            'training.scheduler': 'plateau',
            'training.scheduler_factor': 0.5,
            'training.scheduler_patience': 5
        }.get(key, default)

        optimizer, scheduler = train_script.create_optimizer_and_scheduler(model, config)

        assert isinstance(optimizer, AdamW)
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4

    def test_create_optimizer_unknown(self):
        """Test optimizer creation with unknown optimizer type."""
        import torch.nn as nn

        model = nn.Linear(10, 1)
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            'training.optimizer': 'sgd',  # Not supported
            'training.learning_rate': 0.001,
            'training.weight_decay': 1e-5
        }.get(key, default)

        with pytest.raises(ValueError, match="Unknown optimizer"):
            train_script.create_optimizer_and_scheduler(model, config)

    @patch('scripts.train.logger')
    @patch('scripts.train.sys.exit')
    @patch('scripts.train.load_config')
    def test_main_config_not_found(self, mock_load_config, mock_exit, mock_logger):
        """Test main function with missing config file."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")

        with patch('sys.argv', ['train.py', '--config', '/nonexistent/config.yaml']):
            train_script.main()

        mock_exit.assert_called_with(1)
        mock_logger.error.assert_called()

    @patch('builtins.print')
    @patch('scripts.train.logger')
    @patch('scripts.train.sys.exit')
    @patch('scripts.train.load_config')
    def test_main_eval_only_without_resume(self, mock_load_config, mock_exit,
                                           mock_logger, mock_print):
        """Test main function with eval-only flag but no resume checkpoint."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        test_args = [
            'train.py',
            '--config', '/path/to/config.yaml',
            '--eval-only'
        ]

        with patch('sys.argv', test_args):
            train_script.main()

        mock_exit.assert_called_with(1)
        mock_logger.error.assert_called_with("--eval-only requires --resume")


class TestEvaluateScript:
    """Test cases for evaluation script functions."""

    def test_parse_args_basic(self):
        """Test basic argument parsing for evaluation script."""
        test_args = [
            'evaluate.py',
            '--config', '/path/to/config.yaml',
            '--checkpoint', '/path/to/model.pt'
        ]

        with patch('sys.argv', test_args):
            args = evaluate_script.parse_args()
            assert args.config == Path('/path/to/config.yaml')
            assert args.checkpoint == Path('/path/to/model.pt')
            assert args.data_dir == Path('./data')
            assert args.split == 'test'

    def test_parse_args_all_options(self):
        """Test argument parsing with all options for evaluation."""
        test_args = [
            'evaluate.py',
            '--config', '/path/to/config.yaml',
            '--checkpoint', '/path/to/model.pt',
            '--data-dir', '/custom/data',
            '--output-dir', '/custom/output',
            '--dataset', 'tox21',
            '--split', 'val',
            '--batch-size', '64',
            '--scaffold-analysis',
            '--save-predictions',
            '--plot-results',
            '--device', 'cuda',
            '--debug'
        ]

        with patch('sys.argv', test_args):
            args = evaluate_script.parse_args()
            assert args.dataset == 'tox21'
            assert args.split == 'val'
            assert args.batch_size == 64
            assert args.scaffold_analysis is True
            assert args.save_predictions is True
            assert args.plot_results is True
            assert args.device == 'cuda'
            assert args.debug is True

    def test_toxicity_dataset_with_scaffolds(self):
        """Test ToxicityDataset with scaffold information."""
        mock_graphs = [MagicMock() for _ in range(3)]
        smiles = ['CCO', 'CCN', 'CCC']
        scaffolds = ['scaffold1', 'scaffold2', 'scaffold1']

        dataset = evaluate_script.ToxicityDataset(mock_graphs, smiles, scaffolds)

        assert len(dataset) == 3
        graph, smile, scaffold = dataset[0]
        assert graph == mock_graphs[0]
        assert smile == 'CCO'
        assert scaffold == 'scaffold1'

    def test_toxicity_dataset_without_scaffolds(self):
        """Test ToxicityDataset without scaffold information."""
        mock_graphs = [MagicMock() for _ in range(2)]
        smiles = ['CCO', 'CCN']

        dataset = evaluate_script.ToxicityDataset(mock_graphs, smiles)

        assert len(dataset) == 2
        graph, smile, scaffold = dataset[0]
        assert graph == mock_graphs[0]
        assert smile == 'CCO'
        assert scaffold == ''  # Default empty scaffold

    @patch('scripts.evaluate.torch.load')
    @patch('scripts.evaluate.MultiTaskToxicityPredictor')
    @patch('scripts.evaluate.MoleculeNetLoader')
    def test_load_model_and_data_success(self, mock_loader, mock_model_class, mock_torch_load):
        """Test successful model and data loading."""
        # Mock config
        config = MagicMock()
        config.get.side_effect = lambda key, default=None, required=False: {
            'data.dataset': 'tox21',
            'data.remove_salts': True,
            'data.canonical_smiles': True,
            'data.explicit_h': False,
            'data.use_chirality': True,
            'data.scaffold_split': True,
            'data.train_ratio': 0.8,
            'data.val_ratio': 0.1,
            'data.test_ratio': 0.1,
            'training.random_seed': 42,
            'model.name': 'gcn',
            'model.prediction_hidden_dims': [128, 64],
            'model.prediction_dropout': 0.3,
            'model.use_task_embedding': True
        }.get(key, default)

        # Mock device
        import torch
        with patch('scripts.evaluate.get_device', return_value=torch.device('cpu')):
            # Mock loader
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            import pandas as pd
            mock_df = pd.DataFrame({
                'smiles': ['CCO', 'CCN', 'CCC'],
                'task1': [1, 0, 1],
                'task2': [0, 1, 0]
            })
            mock_loader_instance.load_dataset.return_value = mock_df
            mock_loader_instance.get_task_info.return_value = {
                'task_names': ['task1', 'task2']
            }

            # Mock checkpoint
            mock_checkpoint = {
                'model_state_dict': {'weight': torch.randn(10, 5)},
                'epoch': 10
            }
            mock_torch_load.return_value = mock_checkpoint

            # Mock model
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = Path(temp_dir) / 'model.pt'
                data_dir = Path(temp_dir) / 'data'

                with patch('scripts.evaluate.MoleculePreprocessor'), \
                     patch('scripts.evaluate.GraphFeaturizer'), \
                     patch('scripts.evaluate.ScaffoldSplitter') as mock_splitter:

                    mock_splitter_instance = MagicMock()
                    mock_splitter.return_value = mock_splitter_instance
                    mock_splitter_instance._generate_scaffolds.return_value = ['s1', 's2', 's3']
                    mock_splitter_instance.split.return_value = ([0], [1], [2])

                    # Mock preprocessing
                    with patch('scripts.evaluate.MoleculePreprocessor') as mock_prep, \
                         patch('scripts.evaluate.GraphFeaturizer') as mock_feat:

                        mock_prep_inst = MagicMock()
                        mock_prep.return_value = mock_prep_inst
                        mock_prep_inst.preprocess_smiles.side_effect = lambda x: x

                        mock_feat_inst = MagicMock()
                        mock_feat.return_value = mock_feat_inst
                        mock_graph = MagicMock()
                        mock_feat_inst.featurize.return_value = mock_graph

                        result = evaluate_script.load_model_and_data(
                            config, checkpoint_path, data_dir
                        )

                        model, datasets, task_names, device, train_idx = result

                        assert model == mock_model
                        assert 'train' in datasets
                        assert 'val' in datasets
                        assert 'test' in datasets
                        assert task_names == ['task1', 'task2']
                        assert device == torch.device('cpu')

    @patch('scripts.evaluate.logger')
    @patch('scripts.evaluate.sys.exit')
    @patch('scripts.evaluate.load_config')
    def test_main_config_not_found(self, mock_load_config, mock_exit, mock_logger):
        """Test main function with missing config file."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")

        test_args = [
            'evaluate.py',
            '--config', '/nonexistent/config.yaml',
            '--checkpoint', '/path/to/model.pt'
        ]

        with patch('sys.argv', test_args):
            evaluate_script.main()

        mock_exit.assert_called_with(1)
        mock_logger.error.assert_called()

    @patch('scripts.evaluate.logger')
    @patch('scripts.evaluate.sys.exit')
    def test_main_checkpoint_not_found(self, mock_exit, mock_logger):
        """Test main function with missing checkpoint file."""
        test_args = [
            'evaluate.py',
            '--config', '/path/to/config.yaml',
            '--checkpoint', '/nonexistent/model.pt'
        ]

        with patch('sys.argv', test_args), \
             patch('scripts.evaluate.load_config', return_value=MagicMock()):
            evaluate_script.main()

        mock_exit.assert_called_with(1)
        mock_logger.error.assert_called()

    @patch('scripts.evaluate.json.dump')
    def test_save_results(self, mock_json_dump):
        """Test saving evaluation results."""
        import numpy as np

        predictions = np.array([[0.8, 0.3], [0.2, 0.9]])
        labels = np.array([[1, 0], [0, 1]])
        smiles = ['CCO', 'CCN']
        scaffolds = ['scaffold1', 'scaffold2']
        task_names = ['task1', 'task2']
        metrics = {'auc_roc_mean': 0.85, 'accuracy_mean': 0.80}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            evaluate_script.save_results(
                predictions, labels, smiles, scaffolds,
                task_names, metrics, output_dir
            )

            # Check that files were created
            assert (output_dir / 'predictions.csv').exists()
            assert (output_dir / 'metrics.json').exists()
            assert (output_dir / 'evaluation_report.txt').exists()

            # Verify predictions CSV
            import pandas as pd
            df = pd.read_csv(output_dir / 'predictions.csv')
            assert len(df) == 2
            assert 'smiles' in df.columns
            assert 'task1_pred' in df.columns
            assert 'task2_true' in df.columns


if __name__ == '__main__':
    pytest.main([__file__])