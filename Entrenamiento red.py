# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 15:51:12 2025

@author: Pablo Ramos
"""

import math, time, os, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import pandas as pd
import joblib

import random
from datetime import datetime

torch.backends.cudnn.benchmark = True  # inputs tamaño fijo -> más rápido

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(42)




VRAM_LOG = "torch_vram_log.csv"
def init_vram_log():
    if not Path(VRAM_LOG).exists():
        with open(VRAM_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","stage","epoch","step","alloc_MB","reserved_MB","max_alloc_MB","max_reserved_MB","pid"])

def log_vram(stage, epoch, step):
    if device.type != "cuda": return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserv = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    max_reserv = torch.cuda.max_memory_reserved() / (1024**2)
    with open(VRAM_LOG, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), stage, epoch, step,
                                f"{alloc:.1f}", f"{reserv:.1f}", f"{max_alloc:.1f}", f"{max_reserv:.1f}", os.getpid()])


start = time.time() 


# ======================
#  CARGA DE DATOS
# ======================
data_dir = Path("NN_training")

# Cargar metadatos
X_cols = joblib.load(data_dir / "X_cols.pkl")
y_cols = joblib.load(data_dir / "y_cols.pkl")

# Parquet → DataFrame
train_df = pd.read_parquet(data_dir / "train_scaled.parquet")
val_df   = pd.read_parquet(data_dir / "val_scaled.parquet")
test_df  = pd.read_parquet(data_dir / "test_scaled.parquet")

X_train_s = train_df[X_cols].to_numpy(dtype=np.float32)
y_train_s = train_df[y_cols].to_numpy(dtype=np.float32)

X_val_s = val_df[X_cols].to_numpy(dtype=np.float32)
y_val_s = val_df[y_cols].to_numpy(dtype=np.float32)

X_test_s = test_df[X_cols].to_numpy(dtype=np.float32)
y_test_s = test_df[y_cols].to_numpy(dtype=np.float32)

print("Shapes:",
      X_train_s.shape, y_train_s.shape,
      X_val_s.shape,   y_val_s.shape,
      X_test_s.shape,  y_test_s.shape)

# ======================
#  CONFIG / PREP
# ======================
# Comprobaciones rápidas de dimensiones
n_in  = 11  # (T, P, Y_i)
n_out = 9   # (dY_i de 9 especies)
assert X_train_s.shape[1] == n_in,  f"Esperaba n_in={n_in}, got {X_train_s.shape[1]}"
assert y_train_s.shape[1] == n_out, f"Esperaba n_out={n_out}, got {y_train_s.shape[1]}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets / Tensors
def to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))

train_ds = TensorDataset(to_tensor(X_train_s), to_tensor(y_train_s))
val_ds   = TensorDataset(to_tensor(X_val_s),   to_tensor(y_val_s))
test_ds  = TensorDataset(to_tensor(X_test_s),  to_tensor(y_test_s))

# ======================
#  MODELO
# ======================
class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 1600), nn.GELU(),
            nn.Linear(1600, 800),  nn.GELU(),
            nn.Linear(800, 400),   nn.GELU(),
            nn.Linear(400, n_out)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLP(n_in, n_out).to(device)
criterion = nn.MSELoss(reduction="mean")

# ======================
#  UTILS: EVAL / LOG / CKPT
# ======================
@torch.no_grad()
def evaluate(loader):
    model.eval()
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        bs = xb.size(0)
        loss_sum += loss.item() * bs
        n += bs
    return loss_sum / max(1, n)

def append_csv_row(csv_path, header, row):
    """Append con creación de cabecera si el archivo no existe."""
    new_file = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)

def save_ckpt(ckpt_path, epoch, model, optimizer, scaler, best_val, stage_meta=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
        "stage_meta": stage_meta,
    }
    torch.save(payload, ckpt_path)



def load_ckpt_if_exists(ckpt_path, model, optimizer, scaler):
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        return 1, float("inf"), False  # epoch=1, best_val=inf, not_loaded

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])

    start_epoch = payload.get("epoch", 1)
    best_val = payload.get("best_val", float("inf"))

    # --- Restaurar RNGs ---
    if "rng_cpu" in payload:
        torch.set_rng_state(payload["rng_cpu"])
    if "rng_cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    if "np_rng" in payload:
        np.random.set_state(payload["np_rng"])
    if "py_rng" in payload:
        random.setstate(payload["py_rng"])

    print(f"Resumed from {ckpt_path} at epoch {start_epoch}")
    return start_epoch + 1, best_val, True

# ======================
#  TRAIN STAGE (con logs, ckpt, tqdm)
# ======================
def train_stage(stage_name, train_bs, lr, epochs,
                max_physical_batch=8192, num_workers=2,
                log_csv="training_log.csv",
                ckpt_path=None,
                save_curves_prefix=None,
                resume=True):
    """
    stage_name: str                    -> 'stage1' o 'stage2'
    train_bs: batch lógico objetivo    -> se emula con acumulación si es > max_physical_batch
    lr: learning rate de la etapa
    epochs: épocas de la etapa
    max_physical_batch: batch físico máximo que cabe en GPU
    resume: si True, reanuda desde ckpt si existe
    """
    # DataLoaders
    physical_bs = min(train_bs, max_physical_batch)
    train_loader = DataLoader(train_ds, batch_size=physical_bs, shuffle=True,
                              pin_memory=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=physical_bs, shuffle=False,
                              pin_memory=True, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    accum_steps = max(1, train_bs // physical_bs)

    # Reanudar si procede
    start_epoch = 1
    best_val = float("inf")
    if ckpt_path and resume:
        start_epoch, best_val, _ = load_ckpt_if_exists(ckpt_path, model, optimizer, scaler)

    # Mantener una copia del mejor estado incluso si no hay mejoras nuevas
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Buffers de curvas
    train_curve, val_curve = [], []

    header = ["stage", "epoch", "lr", "train_bs_logical", "train_bs_physical",
              "train_loss", "val_loss", "secs"]
    init_vram_log()

    for epoch in range(start_epoch, epochs+1):
        model.train()
        t0 = time.time()


        optimizer.zero_grad(set_to_none=True)
        steps_since_update = 0
        
        train_loss_sum, train_samples = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{stage_name} | epoch {epoch}/{epochs} | lr={lr:g} | bs={train_bs}", leave=False)
        
        for i, (xb, yb) in enumerate(pbar, start=1):
            xb, yb = xb.to(device), yb.to(device)
        
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                pred = model(xb)
                batch_loss = criterion(pred, yb)
                loss = batch_loss / accum_steps
        
            scaler.scale(loss).backward()
            steps_since_update += 1
        
            # running avg en tqdm
            bs = xb.size(0)
            train_loss_sum += batch_loss.item() * bs
            train_samples  += bs
            pbar.set_postfix({"train_loss": train_loss_sum / max(1, train_samples)})
        
            if steps_since_update == accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0
        
            if i % 50 == 0:
                log_vram(stage_name, epoch, i)
        
        # ---- FLUSH final (si la época no es múltiplo de accum_steps)
        if steps_since_update > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


        # Validación
        val_loss = evaluate(val_loader)
        dur = time.time() - t0
        train_loss_epoch = train_loss_sum / max(1, train_samples)

        log_vram(stage_name, epoch, -1)
    
        # Guardar mejor estado
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # checkpoint inmediato del mejor
            if ckpt_path:
                save_ckpt(ckpt_path, epoch, model, optimizer, scaler, best_val,
                          stage_meta={"stage": stage_name, "lr": lr,
                                     "train_bs": train_bs, "physical_bs": physical_bs})

        # Log en CSV (append)
        append_csv_row(log_csv, header, [
            stage_name, epoch, lr, train_bs, physical_bs,
            f"{train_loss_epoch:.8f}", f"{val_loss:.8f}", f"{dur:.2f}"
        ])

        train_curve.append(train_loss_epoch)
        val_curve.append(val_loss)

        # Print cada 50 épocas (o primera/última)
        if epoch % 50 == 0 or epoch == start_epoch or epoch == epochs:
            print(f"[{stage_name} lr={lr:g}, bs={train_bs}] "
                  f"Epoch {epoch:4d}/{epochs} - train_loss={train_loss_epoch:.6f} "
                  f"- val_loss={val_loss:.6f} - {dur:.1f}s")

    # Restaurar mejor modelo de la etapa si guardamos best_state
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    # Guardar curvas
    if save_curves_prefix:
        np.save(f"{save_curves_prefix}_{stage_name}_train.npy", np.array(train_curve))
        np.save(f"{save_curves_prefix}_{stage_name}_val.npy",   np.array(val_curve))

    return best_val

# ======================
#  EJECUCIÓN: DOS ETAPAS
# ======================
print("===> Stage 1: 25 epochs, batch=1024, lr=1e-4")
best_val_stage1 = train_stage(stage_name="stage1",
                              train_bs=1024, lr=1e-4, epochs=25,
                              max_physical_batch=1024,
                              log_csv="training_log.csv",
                              ckpt_path="checkpoint_stage1.pt",
                              save_curves_prefix="curves",
                              resume=True)

print("\n===> Stage 2: 25 epochs, batch=131072, lr=1e-5")
best_val_stage2 = train_stage(stage_name="stage2",
                              train_bs=131072, lr=1e-5, epochs=25,
                              max_physical_batch=2048,
                              log_csv="training_log.csv",
                              ckpt_path="checkpoint_stage2.pt",
                              save_curves_prefix="curves",
                              resume=True)

# ======================
#  TEST + GUARDADO
# ======================
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, pin_memory=True,num_workers=2)
test_loss = evaluate(test_loader)
print(f"\nFinal test MSE: {test_loss:.6f}")

torch.save(model.state_dict(), "mlp_rde_two_stage.pt")
print("Modelo guardado en 'mlp_rde_two_stage.pt'")

end = time.time()     # Marca final

print(f"Tiempo de ejecución: {end - start:.3f} segundos")
