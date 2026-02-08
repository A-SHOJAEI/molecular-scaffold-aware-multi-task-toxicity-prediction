# Molecular Scaffold-Aware Multi-Task Toxicity Prediction

A hierarchical graph neural network that learns molecular toxicity across 12 Tox21 assays by explicitly modeling scaffold-substructure relationships. The system uses scaffold-aware attention mechanisms to dynamically weight subgraph contributions based on known toxicophore patterns, combined with multi-task learning for simultaneous prediction of all endpoints.

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | Tox21 (7,831 molecules, 12 toxicity endpoints) |
| Model | Scaffold-Aware GCN + Attention Substructure Pooling |
| Backbone | 3-layer GCN, 128 hidden dim |
| Scaffold Dim | 64 |
| Prediction Head | [128, 64] with 0.3 dropout |
| Split Strategy | Scaffold-based (80/10/10) |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Early Stopping | Patience 15, min_delta=0.0001 |
| Training Duration | 19 epochs (early stopping triggered) |
| Batch Size | 32 |
| Seed | 42 |

## Results

### Per-Task AUC-ROC (Scaffold Split)

| Rank | Task | AUC-ROC | Category |
|------|------|---------|----------|
| 1 | NR-AR-LBD | 0.789 | Nuclear Receptor |
| 2 | NR-AhR | 0.729 | Nuclear Receptor |
| 3 | NR-AR | 0.709 | Nuclear Receptor |
| 4 | NR-ER-LBD | 0.695 | Nuclear Receptor |
| 5 | NR-Aromatase | 0.608 | Nuclear Receptor |
| 6 | SR-ATAD5 | 0.594 | Stress Response |
| 7 | SR-HSE | 0.594 | Stress Response |
| 8 | NR-PPAR-gamma | 0.582 | Nuclear Receptor |
| 9 | SR-ARE | 0.564 | Stress Response |
| 10 | NR-ER | 0.555 | Nuclear Receptor |
| 11 | SR-p53 | 0.535 | Stress Response |
| 12 | SR-MMP | 0.526 | Stress Response |

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Mean AUC-ROC | 0.6233 |
| Best Task AUC-ROC | 0.7894 (NR-AR-LBD) |
| Mean Accuracy | 92.47% |

### Analysis

**Scaffold-based splitting is intentionally harder than random splitting.** Unlike random splits that allow structurally similar molecules to appear in both train and test sets, scaffold splitting forces the model to generalize to entirely unseen molecular scaffolds. This evaluates true out-of-distribution generalization rather than memorization. Published Tox21 benchmarks using random splits often report AUC-ROC values of 0.80-0.85+, so the results here reflect a substantially more challenging evaluation protocol.

**Nuclear receptor (NR) tasks consistently outperform stress response (SR) tasks.** The top four tasks are all nuclear receptor assays (mean NR AUC-ROC: 0.667 vs. mean SR AUC-ROC: 0.563). This aligns with the toxicology literature: nuclear receptor binding is more directly determined by molecular scaffold geometry and pharmacophore features that the scaffold-aware architecture explicitly encodes. Stress response pathways involve more indirect, pathway-level mechanisms that are harder to predict from molecular structure alone.

**High accuracy despite moderate AUC-ROC** is a consequence of the severe class imbalance in Tox21 -- most molecules are non-toxic for most endpoints, so a model can achieve high accuracy while the AUC-ROC (which balances sensitivity and specificity) gives a more nuanced picture of discriminative power.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Train scaffold-aware GCN model
python scripts/train.py --config configs/default.yaml

# Evaluate with scaffold analysis
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

## Usage

```python
from molecular_scaffold_aware_multi_task_toxicity_prediction.models.model import MultiTaskToxicityPredictor
from molecular_scaffold_aware_multi_task_toxicity_prediction.data.loader import MoleculeNetLoader

# Load Tox21 dataset
loader = MoleculeNetLoader(data_dir='./data')
df = loader.load_dataset('tox21')

# Create scaffold-aware model
model = MultiTaskToxicityPredictor(
    backbone='gcn',
    backbone_config={'node_dim': 133, 'hidden_dim': 128, 'num_layers': 3},
    num_tasks=12
)
```

## Architecture

The system combines three key components:

1. **Scaffold-Aware Attention**: Multi-head attention mechanism that uses molecular scaffolds as queries to weight substructure contributions
2. **Hierarchical Graph Encoding**: Separate encoders for molecular graphs and scaffold structures with learned fusion
3. **Multi-Task Prediction Head**: Task-specific embeddings with shared backbone for joint toxicity prediction

```
SMILES -> Graph Features -> Scaffold-Aware GNN -> Multi-Task Head -> Toxicity Predictions
              |                    |                    |
          Node/Edge            Attention           Task Embeddings
          Features             Pooling           + Shared Backbone
```

## Project Structure

```
molecular-scaffold-aware-multi-task-toxicity-prediction/
  configs/
    default.yaml              # Training configuration
  scripts/
    train.py                  # Training entry point
    evaluate.py               # Evaluation entry point
  src/
    molecular_scaffold_aware_multi_task_toxicity_prediction/
      data/
        loader.py             # Tox21 data loading
        preprocessing.py      # Molecular graph featurization
      models/
        model.py              # Scaffold-aware GCN model
      training/
        trainer.py            # Training loop with MLflow logging
      evaluation/
        metrics.py            # AUC-ROC, accuracy, per-task metrics
      utils/
        config.py             # YAML config management
  tests/
    test_model.py
    test_data.py
    test_training.py
    test_scripts.py
  notebooks/
    exploration.ipynb
```

## Development

```bash
# Run tests
pytest

# Code quality
black src/ tests/
isort src/ tests/
mypy src/
flake8 src/ tests/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
