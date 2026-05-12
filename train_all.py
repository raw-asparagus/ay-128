"""Train all 11 lab-03 models, one task per process invocation.

Each function below maps to one training run. The orchestrator
``run_all_training.sh`` invokes ``python train_all.py <task>`` in a fresh
Python process per task so GPU memory is fully reclaimed between runs.

Outputs land in ``./artifacts/`` (project root), not ``labs/03/artifacts/``.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "labs" / "03"))

from ugdatalab.utils.compose import Compose
from ugdatalab.models.galaxy_zoo import GalaxyZooGPUDataset
from ugdatalab.models.galaxy_zoo.constants import N_LABELS, LABEL_COLUMNS, LABEL_TREE
from ugdatalab.methods.neural_network.cnn import train_cnn, count_parameters
from ugdatalab.methods.neural_network.augmentation_gpu import GpuCenterCrop, GpuRandomRotation360
from architectures import build_resnet18, build_custom_cnn

ARTIFACTS = PROJECT_ROOT / "artifacts"
INPUT_SIZE = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024


def _load_data():
    img_data = np.load(ARTIFACTS / "galaxy_zoo_images.npz")
    label_data = np.load(ARTIFACTS / "galaxy_zoo_labels.npz")
    split_data = np.load(ARTIFACTS / "split_indices.npz")
    images = img_data["images"]
    labels = label_data["labels"]
    train_idx = split_data["train_idx"]
    val_idx = split_data["val_idx"]
    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx])


def _make_batches(train_x, train_y, val_x, val_y, *, augment):
    train_transform = (
        Compose([GpuRandomRotation360(), GpuCenterCrop(INPUT_SIZE)])
        if augment else Compose([GpuCenterCrop(INPUT_SIZE)])
    )
    val_transform = Compose([GpuCenterCrop(INPUT_SIZE)])
    tb = GalaxyZooGPUDataset(train_x, train_y, batch_size=BATCH_SIZE,
                             transform=train_transform, device=DEVICE, shuffle=True)
    vb = GalaxyZooGPUDataset(val_x, val_y, batch_size=BATCH_SIZE,
                             transform=val_transform, device=DEVICE, shuffle=False)
    return tb, vb


def _adam_fused(params, lr):
    return torch.optim.Adam(params, lr=lr, fused=True)


def _build_custom_from_npz():
    """Load Custom CNN architecture spec from custom_result.npz."""
    data = np.load(ARTIFACTS / "custom_result.npz", allow_pickle=True)
    return build_custom_cnn(
        n_labels=N_LABELS,
        n_channels_list=[int(x) for x in data["n_channels_list"]],
        kernel_sizes=[int(x) for x in data["kernel_sizes"]],
        fc_sizes=[int(x) for x in data["fc_sizes"]],
        dropout_rate=[float(x) for x in np.atleast_1d(data["dropout_rate"])],
        pool_type=str(data["pool_type"]),
        input_size=INPUT_SIZE,
    )


def _slug(name):
    return (name.replace(" ", "_").replace(",", "")
                .replace(".", "p").replace("(", "").replace(")", ""))


# ---------------------------------------------------------------------------
# 03 — ResNet-18, 100 epochs, no aug, no scheduler
# ---------------------------------------------------------------------------

def train_resnet():
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=False)
    model = build_resnet18(n_labels=N_LABELS)
    print(f"ResNet-18 params: {count_parameters(model):,}")
    r = train_cnn(model, tb, vb, n_epochs=100, lr=1e-3, seed=42,
                  optimizer_factory=_adam_fused, scheduler_factory=None)
    torch.save(r.model_state, ARTIFACTS / "resnet18.pt")
    np.savez_compressed(
        ARTIFACTS / "resnet_result.npz",
        train_losses=r.train_losses, val_losses=r.val_losses,
        best_epoch=r.best_epoch, best_val_loss=r.best_val_loss,
        n_parameters=r.n_parameters, learning_rates=r.learning_rates,
    )
    print(f"  best val RMSE: {r.best_val_loss:.4f} at epoch {r.best_epoch+1}")


# ---------------------------------------------------------------------------
# 04a — Architecture sweep, 5 variants × 50 epochs
# ---------------------------------------------------------------------------

ABLATION_SPECS = [
    {"name": "pivot (depth 4, mixed 0.25-0.1)",
     "n_channels_list": [32, 64, 128, 256], "kernel_sizes": [3, 3, 3, 3],
     "fc_sizes": [256, 128], "dropout_rate": [0.25, 0.1], "pool_type": "max"},
    {"name": "depth 3, mixed 0.25-0.1",
     "n_channels_list": [32, 64, 128], "kernel_sizes": [3, 3, 3],
     "fc_sizes": [256, 128], "dropout_rate": [0.25, 0.1], "pool_type": "max"},
    {"name": "depth 5, mixed 0.25-0.1",
     "n_channels_list": [32, 64, 128, 256, 512], "kernel_sizes": [3, 3, 3, 3, 3],
     "fc_sizes": [256, 128], "dropout_rate": [0.25, 0.1], "pool_type": "max"},
    {"name": "depth 4, uniform 0.25",
     "n_channels_list": [32, 64, 128, 256], "kernel_sizes": [3, 3, 3, 3],
     "fc_sizes": [256, 128], "dropout_rate": [0.25, 0.25], "pool_type": "max"},
    {"name": "depth 4, uniform 0.1",
     "n_channels_list": [32, 64, 128, 256], "kernel_sizes": [3, 3, 3, 3],
     "fc_sizes": [256, 128], "dropout_rate": [0.1, 0.1], "pool_type": "max"},
]
ABLATION_EPOCHS = 50


def train_ablation(idx):
    spec = ABLATION_SPECS[idx]
    print(f"  Variant: {spec['name']}")
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=False)
    m = build_custom_cnn(
        n_labels=N_LABELS,
        n_channels_list=spec["n_channels_list"],
        kernel_sizes=spec["kernel_sizes"],
        fc_sizes=spec["fc_sizes"],
        dropout_rate=spec["dropout_rate"],
        pool_type=spec["pool_type"],
        input_size=INPUT_SIZE,
    )
    r = train_cnn(m, tb, vb, n_epochs=ABLATION_EPOCHS, lr=1e-3, seed=42,
                  optimizer_factory=_adam_fused, scheduler_factory=None)
    slug = _slug(spec["name"])
    torch.save(r.model_state, ARTIFACTS / f"ablation_{slug}.pt")
    np.savez_compressed(
        ARTIFACTS / f"ablation_{slug}.npz",
        name=spec["name"],
        train_losses=r.train_losses, val_losses=r.val_losses,
        best_epoch=r.best_epoch, best_val_loss=r.best_val_loss,
        n_parameters=r.n_parameters, learning_rates=r.learning_rates,
        n_channels_list=np.array(spec["n_channels_list"]),
        kernel_sizes=np.array(spec["kernel_sizes"]),
        fc_sizes=np.array(spec["fc_sizes"]),
        dropout_rate=np.array(spec["dropout_rate"]),
        pool_type=spec["pool_type"],
    )
    print(f"  best val RMSE: {r.best_val_loss:.4f} at epoch {r.best_epoch+1}")


def write_ablation_summary():
    names, val_rmse, n_params = [], [], []
    for spec in ABLATION_SPECS:
        slug = _slug(spec["name"])
        data = np.load(ARTIFACTS / f"ablation_{slug}.npz", allow_pickle=True)
        names.append(spec["name"])
        val_rmse.append(float(data["best_val_loss"]))
        n_params.append(int(data["n_parameters"]))
    np.savez_compressed(
        ARTIFACTS / "custom_cnn_ablation.npz",
        names=np.array(names),
        val_rmse=np.asarray(val_rmse),
        n_parameters=np.asarray(n_params),
        n_epochs=ABLATION_EPOCHS,
    )
    print(f"  Saved summary with {len(names)} variants")


# ---------------------------------------------------------------------------
# 04b — Custom CNN baseline, 100 epochs, no aug, no scheduler
# ---------------------------------------------------------------------------

CUSTOM_N_CHANNELS = [32, 64, 128, 256]
CUSTOM_KERNELS = [3, 3, 3, 3]
CUSTOM_FC = [256, 128]
CUSTOM_DROPOUT = [0.25, 0.1]
CUSTOM_POOL = "max"


def train_custom():
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=False)
    m = build_custom_cnn(
        n_labels=N_LABELS,
        n_channels_list=CUSTOM_N_CHANNELS,
        kernel_sizes=CUSTOM_KERNELS,
        fc_sizes=CUSTOM_FC,
        dropout_rate=CUSTOM_DROPOUT,
        pool_type=CUSTOM_POOL,
        input_size=INPUT_SIZE,
    )
    print(f"  Custom CNN params: {count_parameters(m):,}")
    r = train_cnn(m, tb, vb, n_epochs=100, lr=1e-3, seed=42,
                  optimizer_factory=_adam_fused, scheduler_factory=None)
    torch.save(r.model_state, ARTIFACTS / "custom_cnn.pt")
    np.savez_compressed(
        ARTIFACTS / "custom_result.npz",
        train_losses=r.train_losses, val_losses=r.val_losses,
        best_epoch=r.best_epoch, best_val_loss=r.best_val_loss,
        n_parameters=r.n_parameters, learning_rates=r.learning_rates,
        best_model="custom",
        n_channels_list=np.array(CUSTOM_N_CHANNELS),
        kernel_sizes=np.array(CUSTOM_KERNELS),
        fc_sizes=np.array(CUSTOM_FC),
        dropout_rate=np.array(CUSTOM_DROPOUT),
        pool_type=CUSTOM_POOL,
    )
    print(f"  best val RMSE: {r.best_val_loss:.4f} at epoch {r.best_epoch+1}")


# ---------------------------------------------------------------------------
# 05a — Scheduler-only, 50 epochs
# ---------------------------------------------------------------------------

def train_scheduler():
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=False)
    m = _build_custom_from_npz()
    r = train_cnn(m, tb, vb, n_epochs=50, lr=1e-3, seed=42,
                  optimizer_factory=_adam_fused,
                  scheduler_factory=lambda opt: ReduceLROnPlateau(opt, factor=0.5, patience=3))
    np.savez_compressed(
        ARTIFACTS / "scheduler_result.npz",
        train_losses=r.train_losses, val_losses=r.val_losses,
        best_epoch=r.best_epoch, best_val_loss=r.best_val_loss,
        n_parameters=r.n_parameters, learning_rates=r.learning_rates,
    )
    print(f"  best val RMSE: {r.best_val_loss:.4f} at epoch {r.best_epoch+1}")


# ---------------------------------------------------------------------------
# 05b — Augmentation-only, 50 epochs
# ---------------------------------------------------------------------------

def train_aug_only():
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=True)
    m = _build_custom_from_npz()
    r = train_cnn(m, tb, vb, n_epochs=50, lr=1e-3, seed=42,
                  optimizer_factory=_adam_fused, scheduler_factory=None)
    np.savez_compressed(
        ARTIFACTS / "aug_only_result.npz",
        train_losses=r.train_losses, val_losses=r.val_losses,
        best_epoch=r.best_epoch, best_val_loss=r.best_val_loss,
        n_parameters=r.n_parameters, learning_rates=r.learning_rates,
    )
    print(f"  best val RMSE: {r.best_val_loss:.4f} at epoch {r.best_epoch+1}")


# ---------------------------------------------------------------------------
# 05b — Combined (aug + scheduler), 100 epochs — the final committed model
# ---------------------------------------------------------------------------

def train_combined():
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=True)
    m = _build_custom_from_npz()
    r = train_cnn(m, tb, vb, n_epochs=100, lr=1e-3, seed=42,
                  optimizer_factory=_adam_fused,
                  scheduler_factory=lambda opt: ReduceLROnPlateau(opt, factor=0.5, patience=3))
    torch.save(r.model_state, ARTIFACTS / "best_augmented.pt")
    np.savez_compressed(
        ARTIFACTS / "augmented_result.npz",
        train_losses=r.train_losses, val_losses=r.val_losses,
        best_epoch=r.best_epoch, best_val_loss=r.best_val_loss,
        n_parameters=r.n_parameters, learning_rates=r.learning_rates,
    )
    print(f"  best val RMSE: {r.best_val_loss:.4f} at epoch {r.best_epoch+1}")


# ---------------------------------------------------------------------------
# 05c — Tree-weighted RMSE loss, 100 epochs
# ---------------------------------------------------------------------------

def train_treeweighted():
    from tqdm.auto import tqdm
    train_x, train_y, val_x, val_y = _load_data()
    tb, vb = _make_batches(train_x, train_y, val_x, val_y, augment=True)

    # Tree-weighted loss
    name_to_idx = {name: i for i, name in enumerate(LABEL_COLUMNS)}
    parent_indices = [[] for _ in range(N_LABELS)]
    for parent_name, children in LABEL_TREE.items():
        p_idx = name_to_idx[parent_name]
        for child in children:
            parent_indices[name_to_idx[child]].append(p_idx)
    parent_selector = torch.zeros(N_LABELS, N_LABELS)
    for j, parents in enumerate(parent_indices):
        for p in parents:
            parent_selector[j, p] = 1.0
    is_root = torch.tensor([len(p) == 0 for p in parent_indices], dtype=torch.float32)
    EPS = 1e-3

    def tree_weighted_rmse(pred, target):
        sel = parent_selector.to(target.device)
        root = is_root.to(target.device)
        weights = target @ sel.T + root.unsqueeze(0) + EPS
        sq = (pred - target) ** 2
        return torch.sqrt((weights * sq).sum() / weights.sum())

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    m = _build_custom_from_npz().to(DEVICE)
    compiled = torch.compile(m) if DEVICE == "cuda" else m
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, fused=True)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=3)
    n_epochs = 100
    train_losses = np.empty(n_epochs)
    val_losses = np.empty(n_epochs)
    lrs = np.empty(n_epochs)
    best_state, best_val, best_ep = None, float("inf"), 0
    is_cuda = DEVICE == "cuda"

    def _run_epoch(batches, training):
        compiled.train() if training else compiled.eval()
        total, nb = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, targets in batches:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
                    pred = compiled(images)
                    loss = tree_weighted_rmse(pred, targets)
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total += float(loss.item())
                nb += 1
        return total / max(nb, 1)

    for epoch in tqdm(range(n_epochs), desc="Tree-weighted"):
        lrs[epoch] = opt.param_groups[0]["lr"]
        train_losses[epoch] = _run_epoch(tb, training=True)
        val_losses[epoch] = _run_epoch(vb, training=False)
        sched.step(val_losses[epoch])
        if val_losses[epoch] < best_val:
            best_val = float(val_losses[epoch])
            best_ep = epoch
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

    torch.save(best_state, ARTIFACTS / "class_weighted.pt")
    np.savez_compressed(
        ARTIFACTS / "class_weighted_result.npz",
        train_losses=train_losses, val_losses=val_losses,
        best_epoch=best_ep, best_val_loss=best_val,
        n_parameters=count_parameters(m), learning_rates=lrs,
    )
    print(f"  best tree-weighted val RMSE: {best_val:.4f} at epoch {best_ep+1}")


# ---------------------------------------------------------------------------

TASKS = {
    "resnet": train_resnet,
    "ablation_0": lambda: train_ablation(0),
    "ablation_1": lambda: train_ablation(1),
    "ablation_2": lambda: train_ablation(2),
    "ablation_3": lambda: train_ablation(3),
    "ablation_4": lambda: train_ablation(4),
    "ablation_summary": write_ablation_summary,
    "custom": train_custom,
    "scheduler": train_scheduler,
    "aug_only": train_aug_only,
    "combined": train_combined,
    "treeweighted": train_treeweighted,
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=list(TASKS))
    args = p.parse_args()
    print(f"\n=== Task: {args.task} ===")
    t0 = time.time()
    TASKS[args.task]()
    print(f"  ({time.time() - t0:.1f}s)")
