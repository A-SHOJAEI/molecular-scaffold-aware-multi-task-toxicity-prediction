"""Tests for graph neural network models."""

import pytest
import torch
from torch_geometric.data import Batch

from molecular_scaffold_aware_multi_task_toxicity_prediction.models.model import (
    AttentionSubstructurePooling,
    MultiTaskToxicityPredictor,
    ScaffoldAwareGAT,
    ScaffoldAwareGCN,
    ScaffoldAwareGraphSAGE,
)


class TestAttentionSubstructurePooling:
    """Test cases for AttentionSubstructurePooling."""

    def test_init(self):
        """Test AttentionSubstructurePooling initialization."""
        pooling = AttentionSubstructurePooling(
            hidden_dim=64,
            num_heads=4,
            dropout=0.1
        )

        assert pooling.hidden_dim == 64
        assert pooling.num_heads == 4
        assert pooling.dropout.p == 0.1

    def test_forward_without_scaffold(self):
        """Test forward pass without scaffold embedding."""
        pooling = AttentionSubstructurePooling(hidden_dim=32, num_heads=2)

        # Create test data
        node_features = torch.randn(10, 32)  # 10 nodes, 32 features
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])  # 3 graphs

        output = pooling(node_features, batch)

        assert output.shape == torch.Size([3, 32])  # [batch_size, hidden_dim]

    def test_forward_with_scaffold(self):
        """Test forward pass with scaffold embedding."""
        pooling = AttentionSubstructurePooling(hidden_dim=32, num_heads=2)

        node_features = torch.randn(10, 32)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        scaffold_embedding = torch.randn(3, 32)  # 3 graphs

        output = pooling(node_features, batch, scaffold_embedding)

        assert output.shape == torch.Size([3, 32])

    def test_empty_batch(self):
        """Test behavior with empty batch."""
        pooling = AttentionSubstructurePooling(hidden_dim=32)

        node_features = torch.randn(0, 32)
        batch = torch.tensor([], dtype=torch.long)

        output = pooling(node_features, batch)

        assert output.shape == torch.Size([0, 32])


class TestScaffoldAwareGCN:
    """Test cases for ScaffoldAwareGCN."""

    def test_init(self):
        """Test ScaffoldAwareGCN initialization."""
        model = ScaffoldAwareGCN(
            node_dim=133,
            hidden_dim=64,
            num_layers=3,
            dropout=0.2,
            scaffold_dim=32
        )

        assert model.node_dim == 133
        assert model.hidden_dim == 64
        assert model.num_layers == 3
        assert len(model.convs) == 3
        assert len(model.batch_norms) == 3

    def test_forward(self, sample_dataloader):
        """Test forward pass."""
        model = ScaffoldAwareGCN(
            node_dim=133,  # Matching feature dimension
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        )

        for batch in sample_dataloader:
            # Adjust node features to match expected dimension
            if batch.x.size(1) != 133:
                # Pad or truncate to match expected dimension
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            output = model(batch)

            assert output.shape == torch.Size([len(batch.ptr) - 1, 64])
            break  # Test first batch only

    def test_forward_with_scaffold(self, sample_dataloader):
        """Test forward pass with scaffold embeddings."""
        model = ScaffoldAwareGCN(
            node_dim=133,
            hidden_dim=64,
            scaffold_dim=32
        )

        for batch in sample_dataloader:
            # Add scaffold embeddings
            batch_size = len(batch.ptr) - 1
            batch.scaffold_embedding = torch.randn(batch_size, 32)

            # Adjust node features
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            output = model(batch)

            assert output.shape == torch.Size([batch_size, 64])
            break


class TestScaffoldAwareGAT:
    """Test cases for ScaffoldAwareGAT."""

    def test_init(self):
        """Test ScaffoldAwareGAT initialization."""
        model = ScaffoldAwareGAT(
            node_dim=133,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.2
        )

        assert model.node_dim == 133
        assert model.hidden_dim == 64
        assert model.num_layers == 2
        assert model.num_heads == 4

    def test_forward(self, sample_dataloader):
        """Test forward pass."""
        model = ScaffoldAwareGAT(
            node_dim=133,
            hidden_dim=64,
            num_layers=2,
            num_heads=2
        )

        for batch in sample_dataloader:
            # Adjust node features
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            output = model(batch)

            assert output.shape == torch.Size([len(batch.ptr) - 1, 64])
            break


class TestScaffoldAwareGraphSAGE:
    """Test cases for ScaffoldAwareGraphSAGE."""

    def test_init(self):
        """Test ScaffoldAwareGraphSAGE initialization."""
        model = ScaffoldAwareGraphSAGE(
            node_dim=133,
            hidden_dim=64,
            num_layers=2,
            aggr='mean'
        )

        assert model.node_dim == 133
        assert model.hidden_dim == 64
        assert model.num_layers == 2

    def test_forward(self, sample_dataloader):
        """Test forward pass."""
        model = ScaffoldAwareGraphSAGE(
            node_dim=133,
            hidden_dim=64,
            num_layers=2
        )

        for batch in sample_dataloader:
            # Adjust node features
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            output = model(batch)

            assert output.shape == torch.Size([len(batch.ptr) - 1, 64])
            break

    @pytest.mark.parametrize('aggr', ['mean', 'max', 'add'])
    def test_different_aggregation_methods(self, aggr):
        """Test different aggregation methods."""
        model = ScaffoldAwareGraphSAGE(
            node_dim=10,
            hidden_dim=32,
            aggr=aggr
        )

        # Create simple test data
        x = torch.randn(6, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                                  [1, 2, 0, 4, 5, 3]]).long()
        batch = torch.tensor([0, 0, 0, 1, 1, 1])

        data = Batch(x=x, edge_index=edge_index, batch=batch)
        output = model(data)

        assert output.shape == torch.Size([2, 32])


class TestMultiTaskToxicityPredictor:
    """Test cases for MultiTaskToxicityPredictor."""

    def test_init_gcn(self, task_names):
        """Test initialization with GCN backbone."""
        backbone_config = {
            'node_dim': 133,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'scaffold_dim': 32
        }

        model = MultiTaskToxicityPredictor(
            backbone='gcn',
            backbone_config=backbone_config,
            num_tasks=len(task_names),
            hidden_dims=[32, 16],
            dropout=0.2,
            use_task_embedding=True
        )

        assert model.backbone_name == 'gcn'
        assert model.num_tasks == len(task_names)
        assert model.use_task_embedding is True
        assert len(model.task_heads) == len(task_names)

    def test_init_gat(self, task_names):
        """Test initialization with GAT backbone."""
        backbone_config = {
            'node_dim': 133,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'scaffold_dim': 32
        }

        model = MultiTaskToxicityPredictor(
            backbone='gat',
            backbone_config=backbone_config,
            num_tasks=len(task_names)
        )

        assert isinstance(model.backbone, ScaffoldAwareGAT)

    def test_init_sage(self, task_names):
        """Test initialization with GraphSAGE backbone."""
        backbone_config = {
            'node_dim': 133,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'scaffold_dim': 32
        }

        model = MultiTaskToxicityPredictor(
            backbone='sage',
            backbone_config=backbone_config,
            num_tasks=len(task_names)
        )

        assert isinstance(model.backbone, ScaffoldAwareGraphSAGE)

    def test_invalid_backbone(self, task_names):
        """Test error on invalid backbone."""
        backbone_config = {'node_dim': 133, 'hidden_dim': 64}

        with pytest.raises(ValueError, match="Unknown backbone"):
            MultiTaskToxicityPredictor(
                backbone='invalid',
                backbone_config=backbone_config,
                num_tasks=len(task_names)
            )

    def test_forward(self, sample_model, sample_dataloader):
        """Test forward pass."""
        for batch in sample_dataloader:
            # Adjust node features to match model expectation
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            output = sample_model(batch)

            batch_size = len(batch.ptr) - 1
            assert output.shape == torch.Size([batch_size, sample_model.num_tasks])

            # Check output is finite
            assert torch.isfinite(output).all()
            break

    def test_predict_single_task(self, sample_model, sample_dataloader):
        """Test single task prediction."""
        task_id = 0

        for batch in sample_dataloader:
            # Adjust node features
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            output = sample_model.predict_single_task(batch, task_id)

            batch_size = len(batch.ptr) - 1
            assert output.shape == torch.Size([batch_size, 1])
            break

    def test_predict_single_task_invalid_id(self, sample_model, sample_dataloader):
        """Test error on invalid task ID."""
        invalid_task_id = sample_model.num_tasks + 1

        for batch in sample_dataloader:
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            with pytest.raises(ValueError):
                sample_model.predict_single_task(batch, invalid_task_id)
            break

    def test_get_embeddings(self, sample_model, sample_dataloader):
        """Test embedding extraction."""
        for batch in sample_dataloader:
            # Adjust node features
            if batch.x.size(1) != 133:
                current_dim = batch.x.size(1)
                if current_dim < 133:
                    padding = torch.zeros(batch.x.size(0), 133 - current_dim)
                    batch.x = torch.cat([batch.x, padding], dim=1)
                else:
                    batch.x = batch.x[:, :133]

            embeddings = sample_model.get_embeddings(batch)

            batch_size = len(batch.ptr) - 1
            hidden_dim = sample_model.backbone.hidden_dim
            assert embeddings.shape == torch.Size([batch_size, hidden_dim])
            break

    def test_model_without_task_embedding(self, task_names):
        """Test model without task embeddings."""
        backbone_config = {
            'node_dim': 133,
            'hidden_dim': 64,
            'num_layers': 2,
        }

        model = MultiTaskToxicityPredictor(
            backbone='gcn',
            backbone_config=backbone_config,
            num_tasks=len(task_names),
            use_task_embedding=False
        )

        assert model.use_task_embedding is False
        assert not hasattr(model, 'task_embeddings')

    def test_model_training_mode(self, sample_model):
        """Test model training/evaluation mode switching."""
        # Test training mode
        sample_model.train()
        assert sample_model.training is True

        # Test evaluation mode
        sample_model.eval()
        assert sample_model.training is False

    def test_model_parameters(self, sample_model):
        """Test model parameter properties."""
        params = list(sample_model.parameters())
        assert len(params) > 0

        # Check parameters require gradients
        for param in params:
            assert param.requires_grad

        # Count total parameters
        total_params = sum(p.numel() for p in params)
        assert total_params > 0

    @pytest.mark.parametrize('num_tasks', [1, 3, 10])
    def test_different_task_numbers(self, num_tasks):
        """Test models with different numbers of tasks."""
        backbone_config = {
            'node_dim': 50,
            'hidden_dim': 32,
            'num_layers': 2,
        }

        model = MultiTaskToxicityPredictor(
            backbone='gcn',
            backbone_config=backbone_config,
            num_tasks=num_tasks
        )

        assert len(model.task_heads) == num_tasks

        # Test forward pass
        x = torch.randn(10, 50)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).long()
        batch = torch.zeros(10, dtype=torch.long)

        data = Batch(x=x, edge_index=edge_index, batch=batch)
        output = model(data)

        assert output.shape == torch.Size([1, num_tasks])