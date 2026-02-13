"""Custom neural network components for scaffold-aware toxicity prediction.

This module contains specialized components that implement the novel
scaffold-aware attention mechanism and hierarchical graph encoding.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ScaffoldAttentionGate(nn.Module):
    """Gated attention mechanism that modulates substructure features based on scaffold.

    This component implements a learnable gating mechanism that uses scaffold
    information to selectively amplify or suppress molecular substructure features.
    The gate learns which substructures are toxicologically relevant for each
    scaffold type.

    This is a core innovation of the scaffold-aware architecture, enabling
    the model to learn scaffold-specific toxicophore patterns.
    """

    def __init__(self,
                 hidden_dim: int,
                 scaffold_dim: int,
                 dropout: float = 0.1):
        """Initialize scaffold attention gate.

        Args:
            hidden_dim: Dimension of node/substructure features
            scaffold_dim: Dimension of scaffold embeddings
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scaffold_dim = scaffold_dim

        # Gate computation: combines scaffold and substructure features
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + scaffold_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Gate values in [0, 1]
        )

        # Value transformation for gated features
        self.value_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual connection scaling
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self,
                node_features: torch.Tensor,
                scaffold_features: torch.Tensor) -> torch.Tensor:
        """Apply scaffold-gated attention to node features.

        Args:
            node_features: Node features [num_nodes, hidden_dim]
            scaffold_features: Scaffold features [num_nodes, scaffold_dim]
                              (expanded to match node count)

        Returns:
            Gated node features [num_nodes, hidden_dim]
        """
        # Concatenate node and scaffold features
        combined = torch.cat([node_features, scaffold_features], dim=-1)

        # Compute gate values
        gate = self.gate_net(combined)  # [num_nodes, hidden_dim]

        # Transform node features
        transformed = self.value_transform(node_features)

        # Apply gating with residual connection
        gated = gate * transformed + self.residual_scale * node_features

        return gated


class HierarchicalGraphEncoder(nn.Module):
    """Hierarchical encoder that separately processes molecular and scaffold graphs.

    This component encodes molecules at two levels:
    1. Atom-level: Full molecular graph with all atoms and bonds
    2. Scaffold-level: Core structural framework without substituents

    The hierarchical encoding captures both fine-grained atomic details and
    coarse-grained structural patterns, improving generalization across
    molecular scaffolds.
    """

    def __init__(self,
                 node_dim: int,
                 hidden_dim: int,
                 scaffold_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """Initialize hierarchical graph encoder.

        Args:
            node_dim: Dimension of input node features
            hidden_dim: Dimension of hidden features
            scaffold_dim: Dimension of scaffold embeddings
            num_layers: Number of message passing layers
            dropout: Dropout probability
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.scaffold_dim = scaffold_dim
        self.num_layers = num_layers

        # Atom-level encoder
        self.atom_encoder = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            self.atom_encoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        # Scaffold encoder (operates on aggregated features)
        self.scaffold_encoder = nn.Sequential(
            nn.Linear(hidden_dim, scaffold_dim),
            nn.LayerNorm(scaffold_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scaffold_dim, scaffold_dim),
            nn.LayerNorm(scaffold_dim)
        )

        # Cross-level fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + scaffold_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode molecular graph at atom and scaffold levels.

        Args:
            node_features: Input node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch indices [num_nodes]

        Returns:
            Tuple of (fused_features, scaffold_embeddings)
            - fused_features: [num_nodes, hidden_dim]
            - scaffold_embeddings: [batch_size, scaffold_dim]
        """
        # Atom-level encoding
        h = node_features
        for layer in self.atom_encoder:
            h = layer(h)

        # Scaffold-level encoding (pool then encode)
        batch_size = int(batch.max().item()) + 1
        scaffold_emb = []

        for i in range(batch_size):
            mask = (batch == i)
            if mask.any():
                # Pool atom features to get scaffold representation
                graph_h = h[mask].mean(dim=0)  # [hidden_dim]
                scaffold_emb.append(graph_h)

        scaffold_emb = torch.stack(scaffold_emb)  # [batch_size, hidden_dim]
        scaffold_emb = self.scaffold_encoder(scaffold_emb)  # [batch_size, scaffold_dim]

        # Expand scaffold embeddings to node level for fusion
        scaffold_node = scaffold_emb[batch]  # [num_nodes, scaffold_dim]

        # Fuse atom and scaffold features
        fused = self.fusion(torch.cat([h, scaffold_node], dim=-1))

        return fused, scaffold_emb


class ToxicophoreAttention(nn.Module):
    """Attention mechanism that learns to identify toxicophore patterns.

    Toxicophores are molecular substructures associated with toxicity.
    This component learns to attend to potentially toxic substructures
    in a data-driven manner, guided by scaffold context.

    The attention weights can be interpreted as toxicophore relevance scores,
    providing model interpretability.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize toxicophore attention.

        Args:
            hidden_dim: Dimension of features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-head attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Learnable toxicophore query vectors
        self.toxicophore_queries = nn.Parameter(
            torch.randn(num_heads, self.head_dim) * 0.01
        )

    def forward(self,
                node_features: torch.Tensor,
                batch: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """Apply toxicophore attention to identify toxic substructures.

        Args:
            node_features: Node features [num_nodes, hidden_dim]
            batch: Batch indices [num_nodes]
            return_attention: Whether to return attention weights

        Returns:
            Attended features [batch_size, hidden_dim]
            Optional: attention weights if return_attention=True
        """
        batch_size = int(batch.max().item()) + 1
        device = node_features.device

        # Project to queries, keys, values
        q = self.q_proj(node_features)  # [num_nodes, hidden_dim]
        k = self.k_proj(node_features)
        v = self.v_proj(node_features)

        # Reshape for multi-head attention
        # [num_nodes, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        graph_features = []
        attention_weights = [] if return_attention else None

        for i in range(batch_size):
            mask = (batch == i)
            if not mask.any():
                continue

            # Get nodes for this graph
            q_i = q[mask]  # [num_nodes_i, num_heads, head_dim]
            k_i = k[mask]
            v_i = v[mask]

            # Use toxicophore queries
            tox_q = self.toxicophore_queries.unsqueeze(0)  # [1, num_heads, head_dim]

            # Compute attention scores
            scores = torch.einsum('nhd,mhd->nhm', tox_q, k_i)  # [1, num_heads, num_nodes_i]
            scores = scores / (self.head_dim ** 0.5)

            # Softmax over nodes
            attn = F.softmax(scores, dim=-1)  # [1, num_heads, num_nodes_i]
            attn = self.dropout(attn)

            if return_attention:
                attention_weights.append(attn.squeeze(0))  # [num_heads, num_nodes_i]

            # Apply attention to values
            out = torch.einsum('nhm,mhd->nhd', attn, v_i)  # [1, num_heads, head_dim]
            out = out.view(-1, self.hidden_dim)  # [1, hidden_dim]

            graph_features.append(out.squeeze(0))

        graph_features = torch.stack(graph_features)  # [batch_size, hidden_dim]

        # Output projection and residual
        output = self.out_proj(graph_features)
        output = self.layer_norm(output)

        if return_attention:
            return output, attention_weights
        return output


class AdaptiveTaskWeighting(nn.Module):
    """Adaptive weighting for multi-task learning.

    Learns task-specific weights to balance the contribution of each
    toxicity endpoint during training. This addresses the challenge of
    learning from tasks with varying difficulty and class imbalance.

    Implements uncertainty-based weighting similar to Kendall et al. (2018).
    """

    def __init__(self, num_tasks: int):
        """Initialize adaptive task weighting.

        Args:
            num_tasks: Number of toxicity prediction tasks
        """
        super().__init__()
        self.num_tasks = num_tasks

        # Learnable log variance for each task (uncertainty estimation)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            task_losses: Individual task losses [num_tasks]

        Returns:
            Weighted total loss (scalar)
        """
        # Uncertainty weighting: 1/(2*sigma^2) * loss + log(sigma)
        # Where sigma^2 = exp(log_var)
        precision = torch.exp(-self.log_vars)  # 1/sigma^2
        weighted_losses = precision * task_losses + self.log_vars

        return weighted_losses.mean()

    def get_task_weights(self) -> torch.Tensor:
        """Get current task weights for logging.

        Returns:
            Task weights [num_tasks]
        """
        return torch.exp(-self.log_vars).detach()
