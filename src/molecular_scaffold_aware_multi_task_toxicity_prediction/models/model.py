"""Scaffold-aware graph neural network models for molecular toxicity prediction."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

logger = logging.getLogger(__name__)


class AttentionSubstructurePooling(nn.Module):
    """Attention-based substructure pooling for scaffold-aware learning.

    This module computes attention weights over molecular substructures
    and scaffolds to improve generalization to unseen chemical frameworks.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize attention substructure pooling.

        Args:
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Projection layers
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                node_features: torch.Tensor,
                batch: torch.Tensor,
                scaffold_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention pooling.

        Args:
            node_features: Node features [num_nodes, hidden_dim]
            batch: Batch indices [num_nodes]
            scaffold_embedding: Scaffold embeddings [batch_size, hidden_dim]

        Returns:
            Pooled graph features [batch_size, hidden_dim]
        """
        batch_size = int(batch.max().item()) + 1
        device = node_features.device

        logger.debug(f"AttentionSubstructurePooling: batch_size={batch_size}, "
                    f"node_features.shape={node_features.shape}, "
                    f"has_scaffold={scaffold_embedding is not None}")

        # Split node features by batch
        graph_features = []

        for i in range(batch_size):
            # Get nodes for this graph
            mask = (batch == i)
            if not mask.any():
                continue

            graph_nodes = node_features[mask]  # [num_nodes_i, hidden_dim]

            # Use scaffold embedding as query if available
            if scaffold_embedding is not None:
                query = scaffold_embedding[i:i+1]  # [1, hidden_dim]
            else:
                # Use mean node features as query
                query = graph_nodes.mean(dim=0, keepdim=True)  # [1, hidden_dim]

            # Project features
            q = self.query_proj(query)  # [1, hidden_dim]
            k = self.key_proj(graph_nodes)  # [num_nodes_i, hidden_dim]
            v = self.value_proj(graph_nodes)  # [num_nodes_i, hidden_dim]

            # Apply attention
            attn_output, attn_weights = self.attention(
                query=q.unsqueeze(0),  # [1, 1, hidden_dim]
                key=k.unsqueeze(0),    # [1, num_nodes_i, hidden_dim]
                value=v.unsqueeze(0)   # [1, num_nodes_i, hidden_dim]
            )

            # Get pooled feature
            pooled = attn_output.squeeze(0).squeeze(0)  # [hidden_dim]
            graph_features.append(pooled)

        if not graph_features:
            return torch.zeros(0, self.hidden_dim, device=device)

        # Stack and apply output projection
        graph_features = torch.stack(graph_features, dim=0)  # [batch_size, hidden_dim]
        graph_features = self.out_proj(graph_features)
        graph_features = self.dropout(graph_features)
        graph_features = self.layer_norm(graph_features)

        return graph_features


class ScaffoldAwareGCN(nn.Module):
    """Scaffold-aware Graph Convolutional Network.

    Incorporates scaffold information through attention-based pooling
    and scaffold-aware node embeddings.
    """

    def __init__(self,
                 node_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 scaffold_dim: int = 64):
        """Initialize scaffold-aware GCN.

        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            dropout: Dropout probability
            scaffold_dim: Scaffold embedding dimension
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaffold_dim = scaffold_dim

        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # Scaffold projection
        self.scaffold_proj = nn.Linear(scaffold_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Attention pooling
        self.attention_pool = AttentionSubstructurePooling(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            data: Batch of molecular graphs

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:  # No dropout after last layer
                x = self.dropout(x)

        # Prepare scaffold embeddings
        scaffold_embedding = None
        if hasattr(data, 'scaffold_embedding'):
            scaffold_embedding = self.scaffold_proj(data.scaffold_embedding)

        # Attention pooling
        graph_embedding = self.attention_pool(x, batch, scaffold_embedding)

        return graph_embedding


class ScaffoldAwareGAT(nn.Module):
    """Scaffold-aware Graph Attention Network.

    Uses attention mechanisms both in message passing and in
    scaffold-aware pooling for enhanced molecular representation.
    """

    def __init__(self,
                 node_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 scaffold_dim: int = 64):
        """Initialize scaffold-aware GAT.

        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            scaffold_dim: Scaffold embedding dimension
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.scaffold_dim = scaffold_dim

        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # Scaffold projection
        self.scaffold_proj = nn.Linear(scaffold_dim, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                conv = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
            else:
                conv = GATConv(
                    hidden_dim, hidden_dim // num_heads,
                    heads=num_heads, dropout=dropout
                )
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Attention pooling
        self.attention_pool = AttentionSubstructurePooling(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            data: Batch of molecular graphs

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        # Prepare scaffold embeddings
        scaffold_embedding = None
        if hasattr(data, 'scaffold_embedding'):
            scaffold_embedding = self.scaffold_proj(data.scaffold_embedding)

        # Attention pooling
        graph_embedding = self.attention_pool(x, batch, scaffold_embedding)

        return graph_embedding


class ScaffoldAwareGraphSAGE(nn.Module):
    """Scaffold-aware GraphSAGE Network.

    Combines GraphSAGE sampling with scaffold-aware pooling
    for scalable molecular property prediction.
    """

    def __init__(self,
                 node_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 scaffold_dim: int = 64,
                 aggr: str = 'mean'):
        """Initialize scaffold-aware GraphSAGE.

        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of SAGE layers
            dropout: Dropout probability
            scaffold_dim: Scaffold embedding dimension
            aggr: Aggregation method ('mean', 'max', 'add')
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaffold_dim = scaffold_dim

        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # Scaffold projection
        self.scaffold_proj = nn.Linear(scaffold_dim, hidden_dim)

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            conv = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Attention pooling
        self.attention_pool = AttentionSubstructurePooling(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            data: Batch of molecular graphs

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        x = self.dropout(x)

        # GraphSAGE layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        # Prepare scaffold embeddings
        scaffold_embedding = None
        if hasattr(data, 'scaffold_embedding'):
            scaffold_embedding = self.scaffold_proj(data.scaffold_embedding)

        # Attention pooling
        graph_embedding = self.attention_pool(x, batch, scaffold_embedding)

        return graph_embedding


class MultiTaskToxicityPredictor(nn.Module):
    """Multi-task neural network for toxicity prediction.

    Combines scaffold-aware graph neural networks with multi-task
    prediction heads for simultaneous prediction of multiple toxicity endpoints.
    """

    def __init__(self,
                 backbone: str,
                 backbone_config: Dict,
                 num_tasks: int,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.3,
                 use_task_embedding: bool = True):
        """Initialize multi-task toxicity predictor.

        Args:
            backbone: Backbone architecture ('gcn', 'gat', 'sage')
            backbone_config: Configuration for backbone model
            num_tasks: Number of toxicity prediction tasks
            hidden_dims: Hidden layer dimensions for prediction heads
            dropout: Dropout probability
            use_task_embedding: Whether to use task-specific embeddings
        """
        super().__init__()
        self.backbone_name = backbone
        self.num_tasks = num_tasks
        self.use_task_embedding = use_task_embedding

        # Initialize backbone
        if backbone == 'gcn':
            self.backbone = ScaffoldAwareGCN(**backbone_config)
        elif backbone == 'gat':
            self.backbone = ScaffoldAwareGAT(**backbone_config)
        elif backbone == 'sage':
            self.backbone = ScaffoldAwareGraphSAGE(**backbone_config)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        backbone_dim = backbone_config['hidden_dim']

        # Task embeddings for multi-task learning
        if use_task_embedding:
            self.task_embeddings = nn.Embedding(num_tasks, backbone_dim)
            prediction_input_dim = backbone_dim * 2
        else:
            prediction_input_dim = backbone_dim

        # Shared prediction layers
        prediction_layers = []
        prev_dim = prediction_input_dim

        for hidden_dim in hidden_dims:
            prediction_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*prediction_layers)

        # Task-specific prediction heads
        self.task_heads = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in range(num_tasks)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self,
                data: Batch,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for multi-task prediction.

        Args:
            data: Batch of molecular graphs
            task_ids: Task indices for task-specific prediction

        Returns:
            Predictions for all tasks [batch_size, num_tasks]
        """
        # Get graph embeddings from backbone
        graph_embedding = self.backbone(data)  # [batch_size, hidden_dim]
        batch_size = graph_embedding.size(0)

        # Prepare predictions for all tasks
        all_predictions = []

        for task_id in range(self.num_tasks):
            # Prepare input features
            if self.use_task_embedding:
                task_emb = self.task_embeddings(
                    torch.full((batch_size,), task_id,
                              device=graph_embedding.device, dtype=torch.long)
                )
                task_input = torch.cat([graph_embedding, task_emb], dim=1)
            else:
                task_input = graph_embedding

            # Apply shared layers
            features = self.shared_layers(task_input)

            # Apply task-specific head
            prediction = self.task_heads[task_id](features)
            all_predictions.append(prediction)

        # Stack predictions
        predictions = torch.cat(all_predictions, dim=1)  # [batch_size, num_tasks]

        return predictions

    def predict_single_task(self,
                           data: Batch,
                           task_id: int) -> torch.Tensor:
        """Predict for a single task.

        Args:
            data: Batch of molecular graphs
            task_id: Task index

        Returns:
            Task-specific predictions [batch_size, 1]
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task ID {task_id} >= num_tasks {self.num_tasks}")

        graph_embedding = self.backbone(data)
        batch_size = graph_embedding.size(0)

        if self.use_task_embedding:
            task_emb = self.task_embeddings(
                torch.full((batch_size,), task_id,
                          device=graph_embedding.device, dtype=torch.long)
            )
            task_input = torch.cat([graph_embedding, task_emb], dim=1)
        else:
            task_input = graph_embedding

        features = self.shared_layers(task_input)
        prediction = self.task_heads[task_id](features)

        return prediction

    def get_embeddings(self, data: Batch) -> torch.Tensor:
        """Get graph embeddings without prediction.

        Args:
            data: Batch of molecular graphs

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        return self.backbone(data)