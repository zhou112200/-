import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Optional: resume from checkpoint.
RESUME_PATH = None

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

dtype = torch.float16  # T4: fp16 is faster than bf16
pin_memory = device == "cuda"
num_workers = 2


def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(2025)

# ======================================================
# 1. Data loading
# ======================================================
DATA_DIR = "/content"

train_ids = np.load(f"{DATA_DIR}/train.npy", mmap_mode="r")
val_ids = np.load(f"{DATA_DIR}/val.npy", mmap_mode="r")
with open(f"{DATA_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

vocab_size = meta["vocab_size"]
block_size = 256
batch_size = 20
grad_accum = 2


class FastGPTIterable(IterableDataset):
    """Infinite stream sampling without dataset reconstruction."""

    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data
        self.block_size = block_size
        self.max_start = len(data) - block_size - 1

    def __iter__(self):
        while True:
            i = random.randint(0, self.max_start)
            x = torch.from_numpy(
                self.data[i : i + self.block_size].astype(np.int64)
            )
            y = torch.from_numpy(
                self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)
            )
            yield x, y


def build_train_loader(data: np.ndarray) -> DataLoader:
    return DataLoader(
        FastGPTIterable(data, block_size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )


def build_val_loader(data: np.ndarray) -> DataLoader:
    return DataLoader(
        FastGPTIterable(data, block_size),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


train_loader = build_train_loader(train_ids)
val_loader = build_val_loader(val_ids)

# ======================================================
# 2. Model definition
# ======================================================


class GPTConfig:
    def __init__(
        self,
        block_size,
        vocab_size,
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 384,
        dropout: float = 0.1,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class SDPACausalAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embd = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(bsz, seq_len, embd)
        out = self.proj(out)
        return self.attn_drop(out)


class FusedBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SDPACausalAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_drop(self.attn(self.ln1(x)))
        x = x + self.resid_drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        self.blocks = nn.ModuleList(
            [FusedBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.head.weight = self.tok_emb.weight  # tie weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq_len = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :seq_len, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


# ======================================================
# 3. Model setup
# ======================================================
config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=8,
    n_head=8,
    n_embd=384,
    dropout=0.1,
)

model = GPT(config).to(device)
ENABLE_COMPILE = False
if ENABLE_COMPILE and hasattr(torch, "compile"):
    model = torch.compile(model)

# ======================================================
# 4. Optimizer & scheduler
# ======================================================
decay_params = []
no_decay_params = []

for _, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.dim() >= 2:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

use_fused = torch.cuda.is_available()
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=6e-4,
    betas=(0.9, 0.999),
    fused=use_fused,
)

scaler = torch.cuda.amp.GradScaler()

max_steps = 6000
warmup_steps = 200


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    return 0.1 + 0.9 * 0.5 * (
        1.0
        + math.cos(
            math.pi * (step - warmup_steps)
            / max(max_steps - warmup_steps, 1)
        )
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ======================================================
# 5. Resume logic
# ======================================================
start_step = 0
current_val_loss = None

if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"Loading checkpoint from {RESUME_PATH} ...")
    checkpoint = torch.load(RESUME_PATH, map_location=device)

    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_step = int(checkpoint.get("step", 0))
    current_val_loss = checkpoint.get("val_loss", None)
    print(f"Resuming training from step {start_step}")

# ======================================================
# 6. Training loop
# ======================================================
eval_interval = 500
save_interval = 1000
clip_grad = 1.0

train_iter = iter(train_loader)
pbar = tqdm(total=max_steps, initial=start_step)

for step in range(start_step, max_steps):
    model.train()
    total_loss = 0.0

    for _ in range(grad_accum):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(xb, yb)
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        total_loss += float(loss.detach())

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    if (step + 1) % 200 == 0:
        pbar.set_postfix(
            {
                "loss": f"{total_loss * grad_accum:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            }
        )

    if (step + 1) % eval_interval == 0:
        model.eval()
        val_losses: list[float] = []

        with torch.inference_mode():
            val_iter = iter(val_loader)
            for _ in range(8):
                try:
                    xv, yv = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    xv, yv = next(val_iter)

                xv = xv.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)

                with torch.autocast(device_type=device, dtype=dtype):
                    _, l = model(xv, yv)
                val_losses.append(l.item())

        current_val_loss = float(sum(val_losses) / len(val_losses))
        print(f"\n[Eval] step {step+1} | val loss {current_val_loss:.4f}")
        model.train()

    if (step + 1) % save_interval == 0:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step + 1,
            "config": config.__dict__,
            "scaler": scaler.state_dict(),
            "val_loss": current_val_loss,
        }
        if current_val_loss is not None:
            save_name = f"ckpt_step{step+1}_val{current_val_loss:.4f}.pt"
        else:
            save_name = f"ckpt_step{step+1}.pt"

        torch.save(ckpt, save_name)
        print(f"\nSaved checkpoint: {save_name}")

    pbar.update(1)

pbar.close()

final_ckpt = {
    "model": model.state_dict(),
    "config": config.__dict__,
    "step": max_steps,
    "val_loss": current_val_loss,
}
if current_val_loss is not None:
    final_name = f"final_model_val{current_val_loss:.4f}.pt"
else:
    final_name = "final_model.pt"

torch.save(final_ckpt, final_name)
print("Training complete!")
