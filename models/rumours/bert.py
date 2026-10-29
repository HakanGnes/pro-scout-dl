"""BERTurk fine-tuned to read a masked headline and say whether the claimed move happens."""
from __future__ import annotations

import math
import time

import numpy as np
from sklearn.metrics import roc_auc_score

MODEL_NAME = "dbmdz/bert-base-turkish-uncased"   # slugs are lower-case and ASCII-folded
HPARAMS = dict(max_len=64, batch_size=32, lr=2e-5, weight_decay=0.01, warmup=0.1, epochs=4)


def _device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def predict(model, tok, texts: list[str], device, max_len: int, batch: int = 128) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**enc).logits[:, 0]
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fine_tune(train_texts: list[str], train_y: np.ndarray, val_texts: list[str], val_y: np.ndarray,
              predict_on: dict[str, list[str]], seed: int, model_name: str = MODEL_NAME, device: str = "auto",
              hparams: dict | None = None, log=print) -> tuple[dict[str, np.ndarray], dict]:
    """One seed: train for the set number of epochs, keep the epoch with the best validation AUC."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    hp = {**HPARAMS, **(hparams or {})}
    dev = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps = hp["epochs"] * math.ceil(len(train_texts) / hp["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    y = torch.tensor(train_y, dtype=torch.float32)
    best, best_auc, history = None, -1.0, []
    for epoch in range(hp["epochs"]):
        model.train()
        started = time.perf_counter()
        order = rng.permutation(len(train_texts))
        for i in range(0, len(order), hp["batch_size"]):
            idx = order[i:i + hp["batch_size"]]
            enc = tok([train_texts[j] for j in idx], truncation=True, max_length=hp["max_len"], padding=True,
                      return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(**enc).logits[:, 0]
            loss = loss_fn(logits.float(), y[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv = predict(model, tok, val_texts, dev, hp["max_len"])
        auc = float(roc_auc_score(val_y, pv)) if len(set(val_y)) > 1 else float("nan")
        history.append(round(auc, 4))
        log(f"    seed {seed} epoch {epoch + 1}: val AUC {auc:.4f} ({time.perf_counter() - started:.0f}s)")
        if best is None or auc > best_auc:
            best_auc = auc
            best = {k: predict(model, tok, t, dev, hp["max_len"]) for k, t in predict_on.items()}
            best_epoch = epoch + 1
    return best, {"seed": seed, "val_auc_by_epoch": history, "best_epoch": best_epoch, "device": dev.type}
