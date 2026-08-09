"""Quick diagnostic: trace through experiment setup one step at a time."""
import sys, time
sys.path.insert(0, '.')
import torch

print("1. Loading configs...")
from src import config as cfgmod, data_prep, paths
base = cfgmod.load_base_config('configs/base_config.yaml')
exps = cfgmod.load_experiments('configs/experiments.yaml')
cfg = cfgmod.resolve_experiment(base, exps[0])
print(f"   experiment: {cfg['name']}")

print("2. Loading artifacts...")
manifest, norm_stats = data_prep.load_artifacts(base)
print(f"   manifest rows: {len(manifest)}, norm channels: {list(norm_stats.keys())}")

print("3. Creating model + moving to CUDA...")
device = torch.device("cuda")
from src.models import create_model
model = create_model(cfg).to(device)
print(f"   model created OK")

print("4. Creating loss + scaler...")
from src.losses import create_loss
criterion = create_loss(cfg).to(device)
scaler = torch.amp.GradScaler("cuda", enabled=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"   loss/scaler OK")

print("5. Building dataset (train) + SceneGroupedSampler...")
from src.dataset import RadarPatchDataset, SceneGroupedSampler
train_ds = RadarPatchDataset(cfg, manifest, "train", norm_stats, mode="train")
sampler = SceneGroupedSampler(train_ds, shuffle=True, seed=42)
sampler.set_epoch(0)
print(f"   dataset len={len(train_ds)}, sampler len={len(sampler)}")

print("6. Building DataLoader...")
from torch.utils.data import DataLoader
loader = DataLoader(train_ds, batch_size=8, sampler=sampler, drop_last=True, num_workers=0, pin_memory=False)
print(f"   loader batches={len(loader)}")

print("7. Timing first 20 batches (should be fast after first scene load)...")
t0 = time.time()
for i, (x, y) in enumerate(loader):
    if i == 0:
        print(f"   batch 0 loaded: {time.time()-t0:.2f}s  shape={x.shape}")
    if i == 4:
        print(f"   batch 4 (end of scene 0): {time.time()-t0:.2f}s")
    if i == 5:
        print(f"   batch 5 (scene 1 start): {time.time()-t0:.2f}s")
    if i == 19:
        print(f"   batch 19: {time.time()-t0:.2f}s")
        break

print("8. Timing 1 full epoch (moving to GPU)...")
t0 = time.time()
n_batches = 0
sampler.set_epoch(0)
for x, y in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", enabled=True):
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    n_batches += 1
    if n_batches % 100 == 0:
        print(f"   {n_batches}/{len(loader)} batches, {time.time()-t0:.1f}s elapsed")
print(f"   full epoch done: {n_batches} batches in {time.time()-t0:.1f}s")
