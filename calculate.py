import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# config
# =========================
SEED = 0

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures_part2")

SAVE_PATH_BEST = os.path.join(OUT_DIR, "resnet18_cifar100_best.pt")
SAVE_SPLITS = os.path.join(OUT_DIR, "cifar100_splits.pt")
SAVE_TRAIN_FEATS = os.path.join(OUT_DIR, "train_penultimate_feats_labels.pt")
SAVE_NECO_PARAMS = os.path.join(OUT_DIR, "neco_params.pt")
SAVE_CURVES = os.path.join(OUT_DIR, "training_curves.pt")

MINIBATCH_SIZE = 512
NUM_CLASSES = 100

RUN_OOD_SCORES = True
RUN_NC_1_TO_5 = True
RUN_NC_ACROSS_LAYERS = True

OOD_NAME = "SVHN"


# =========================
# setup
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type == "cuda")
PIN = (device.type == "cuda")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =========================
# model (keep identical to checkpoint)
# =========================
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c, 1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + identity
        return F.relu(y)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_c, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_c, out_c, stride=s))
            self.in_c = out_c * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.layer4(self.layer3(self.layer2(self.layer1(y))))
        y = F.adaptive_avg_pool2d(y, 1)
        y = y.view(y.size(0), -1)
        return self.fc(y)


def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def load_checkpoint(path, model, map_location):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint: {path}")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    print("loaded:", path)
    return ckpt


# =========================
# data
# =========================
def build_cifar100_loaders():
    if not os.path.exists(SAVE_SPLITS):
        raise FileNotFoundError(f"missing splits file: {SAVE_SPLITS}")

    splits = torch.load(SAVE_SPLITS, map_location="cpu")
    mean = splits["mean"]
    std = splits["std"]

    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = datasets.CIFAR100(DATA_DIR, train=True, download=False, transform=tf)
    test_full = datasets.CIFAR100(DATA_DIR, train=False, download=False, transform=tf)

    # case A: train/val indices defined on train set
    if "train_indices" in splits and "val_indices" in splits:
        train_ds_eval = torch.utils.data.Subset(train_full, splits["train_indices"])
        val_ds = torch.utils.data.Subset(train_full, splits["val_indices"])
        test_ds = test_full

    # case B: val/test indices defined on official test set
    elif "val_indices" in splits and "test_indices" in splits:
        val_ds = torch.utils.data.Subset(test_full, splits["val_indices"])
        test_ds = torch.utils.data.Subset(test_full, splits["test_indices"])
        train_ds_eval = train_full

    else:
        raise RuntimeError(
            "SAVE_SPLITS format not recognized. Expected "
            "('train_indices' & 'val_indices') OR ('val_indices' & 'test_indices')."
        )

    train_loader_eval = DataLoader(
        train_ds_eval, batch_size=MINIBATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN
    )
    test_loader = DataLoader(
        test_ds, batch_size=MINIBATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN
    )

    # val_loader is optional (only useful if you explicitly need it)
    val_loader = DataLoader(
        val_ds, batch_size=MINIBATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN
    )

    return train_loader_eval, val_loader, test_loader, mean, std


def build_ood_loader(mean, std):
    ood_tf = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize(mean, std)])

    if OOD_NAME.upper() == "SVHN":
        ood_ds = datasets.SVHN(root=DATA_DIR, split="test", download=False, transform=ood_tf)
    else:
        raise ValueError(f"unsupported OOD_NAME={OOD_NAME}")

    return DataLoader(ood_ds, batch_size=MINIBATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN)


# =========================
# feature extraction
# =========================
@torch.no_grad()
def extract_logits_features(model, loader):
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    feats_buf = {}

    def hook_fc_input(_, inp, __):
        feats_buf["feats"] = inp[0].detach()

    h = model.fc.register_forward_hook(hook_fc_input)

    for x, y in loader:
        feats_buf.clear()
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(x)
        feats = feats_buf["feats"]
        all_logits.append(logits.detach().cpu())
        all_feats.append(feats.detach().cpu())
        all_labels.append(y.detach().cpu())

    h.remove()
    return torch.cat(all_logits), torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def extract_layer_features_stream(model, loader, layer_names=("layer1", "layer2", "layer3", "layer4")):
    model.eval()
    buf = {}

    def make_hook(name):
        def hook(_, __, out):
            buf[name] = out.detach()
        return hook

    handles = [getattr(model, name).register_forward_hook(make_hook(name)) for name in layer_names]

    for x, y in loader:
        buf.clear()
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _ = model(x)

        feats_batch = {}
        for name in layer_names:
            z = buf[name]  # [B,C,H,W] on GPU
            z = F.adaptive_avg_pool2d(z, 1).flatten(1)  # [B,D]
            feats_batch[name] = z

        yield feats_batch, y  # y on CPU

    for h in handles:
        h.remove()


# =========================
# OOD scores + metrics
# =========================
def score_msp(logits):
    return F.softmax(logits, dim=1).max(dim=1).values


def score_maxlogit(logits):
    return logits.max(dim=1).values


def score_energy(logits, temp=1.0):
    return temp * torch.logsumexp(logits / temp, dim=1)


def fit_mahalanobis(train_feats, train_labels, num_classes):
    train_feats = train_feats.float()
    train_labels = train_labels.long()
    n, d = train_feats.shape

    class_means = torch.zeros(num_classes, d)
    for c in range(num_classes):
        fc = train_feats[train_labels == c]
        class_means[c] = fc.mean(dim=0)

    centered = train_feats - class_means[train_labels]
    cov = (centered.T @ centered) / (n - num_classes)

    cov = cov + 1e-4 * torch.eye(d, dtype=cov.dtype, device=cov.device)
    precision = torch.linalg.inv(cov)
    return class_means, precision


def score_mahalanobis(feats, class_means, precision):
    feats = feats.float()
    diff = feats.unsqueeze(1) - class_means.unsqueeze(0)
    d2 = torch.einsum("ncd,dd,ncd->nc", diff, precision, diff)
    return -d2.min(dim=1).values


def fit_vim(train_feats, W, r_dim=64):
    train_feats = train_feats.float()
    W = W.float()
    u = W.mean(dim=0)

    X = train_feats - u
    _, _, Vt = torch.linalg.svd(W, full_matrices=False)
    V = Vt.T
    P = V @ V.T

    R = X - X @ P
    _, _, Vt_r = torch.linalg.svd(R, full_matrices=False)
    Vr = Vt_r[:r_dim].T
    return {"u": u, "P": P, "Vr": Vr}


def score_vim(logits, feats, vim_params):
    u, P, Vr = vim_params["u"], vim_params["P"], vim_params["Vr"]
    feats = feats.float()
    X = feats - u
    R = X - X @ P
    z = R @ Vr
    residual_norm = torch.norm(z, dim=1)
    maxlogit = logits.max(dim=1).values
    return maxlogit - residual_norm


def score_neco(logits, feats, neco_params, eps=1e-12):
    mu = neco_params["mu"]
    Vd = neco_params["Vd"]
    X = feats.float() - mu.unsqueeze(0)
    proj = X @ Vd
    ratio = torch.norm(proj, dim=1) / (torch.norm(X, dim=1) + eps)
    maxlogit = logits.max(dim=1).values.float()
    return ratio * maxlogit


def auroc(id_scores, ood_scores):
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(s)) + 1
    pos_ranks = ranks[y == 1]
    n_pos = len(pos_ranks)
    n_neg = len(s) - n_pos
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aupr(id_scores, ood_scores):
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))]).astype(np.int64)
    s = np.concatenate([id_scores, ood_scores])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    return float(np.trapz(precision, recall))


def fpr_at_95_tpr(id_scores, ood_scores):
    thr = np.quantile(np.asarray(id_scores), 0.05)
    return float((np.asarray(ood_scores) >= thr).mean())


def summarize_method(name, id_s, ood_s):
    out = {
        "AUROC": auroc(id_s, ood_s),
        "AUPR": aupr(id_s, ood_s),
        "FPR@95TPR": fpr_at_95_tpr(id_s, ood_s),
    }
    print(f"{name:12s} | AUROC {out['AUROC']:.4f} | AUPR {out['AUPR']:.4f} | FPR@95 {out['FPR@95TPR']:.4f}")
    return out


def save_hist(id_scores, ood_scores, title, fname):
    plt.figure(figsize=(6, 4))
    plt.hist(id_scores, bins=80, alpha=0.6, label="ID")
    plt.hist(ood_scores, bins=80, alpha=0.6, label="OOD")
    plt.title(title)
    plt.xlabel("score (higher = more ID)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    print("saved fig:", path)


# =========================
# Neural Collapse (penultimate + helpers)
# =========================
@torch.no_grad()
def compute_class_stats(feats, labels, num_classes):
    feats = feats.float()
    labels = labels.long()
    means = torch.zeros(num_classes, feats.shape[1])
    counts = torch.zeros(num_classes, dtype=torch.long)
    for c in range(num_classes):
        idx = (labels == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            means[c] = feats[idx].mean(dim=0)
    diffs = feats - means[labels]
    within_var = (diffs.pow(2).sum(dim=1)).mean().item()
    return means, within_var, counts


def gram_cosine_matrix(vectors):
    v = vectors.float()
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
    return v @ v.T


def pairwise_mean_distances(means):
    diff = means.unsqueeze(1) - means.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    C = means.size(0)
    offdiag = dists[~torch.eye(C, dtype=torch.bool, device=dists.device)]
    return offdiag.cpu().numpy()


def nc1_metric(within_var, means):
    global_mean = means.mean(dim=0)
    between_var = (means - global_mean).pow(2).sum(dim=1).mean().item()
    return within_var / (between_var + 1e-12), between_var


def nc2_metric(means):
    G = gram_cosine_matrix(means)
    C = G.size(0)
    I = torch.eye(C, dtype=torch.bool, device=G.device)
    off = G[~I]
    return off.mean().item(), off.std().item()


def nc3_metric(W, means):
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-12)
    Mn = means / (means.norm(dim=1, keepdim=True) + 1e-12)
    cos = (Wn * Mn).sum(dim=1)
    return cos.mean().item(), cos.std().item(), cos.cpu().numpy()


def nc4_metric(W):
    G = gram_cosine_matrix(W)
    C = G.size(0)
    I = torch.eye(C, dtype=torch.bool, device=G.device)
    off = G[~I]
    return off.mean().item(), off.std().item()


def nc5_metric(means):
    means = means.float()
    C, _ = means.shape
    M = means - means.mean(dim=0, keepdim=True)
    M = M / (M.norm(dim=1, keepdim=True) + 1e-12)
    G = (M @ M.T).cpu().numpy()

    target = -1.0 / (C - 1)
    I = np.eye(C, dtype=bool)
    off = G[~I]

    abs_dev_mean = np.mean(np.abs(off - target))
    G_etf = np.eye(C) + target * (np.ones((C, C)) - np.eye(C))
    fro_dev = np.linalg.norm(G - G_etf, ord="fro") / C
    eigs = np.linalg.eigvalsh(G)

    return {
        "target": float(target),
        "abs_dev_mean": float(abs_dev_mean),
        "fro_dev": float(fro_dev),
        "eigs": eigs,
        "gram": G,
    }


def save_nc_plots(offdiag_dists, within_var, between_var, nc3_per_class, nc5, means, W):
    # NC1 parts
    ratio = within_var / (between_var + 1e-12)
    plt.figure(figsize=(4, 4))
    plt.bar([0, 1], [within_var, between_var])
    plt.xticks([0, 1], ["within", "between"])
    plt.title(f"NC1 parts (ratio={ratio:.4f})")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc1_within_vs_between.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # distances between class means
    plt.figure(figsize=(6, 4))
    plt.hist(offdiag_dists, bins=60)
    plt.title("class mean distances (off-diagonal)")
    plt.xlabel("euclidean distance")
    plt.ylabel("count")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc_class_mean_dist_hist.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # within var alone
    plt.figure(figsize=(4, 4))
    plt.bar([0], [within_var])
    plt.title("within-class variance (train, penultimate)")
    plt.xticks([0], ["train"])
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc_within_var.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # NC3 per class
    plt.figure(figsize=(6, 4))
    plt.hist(nc3_per_class, bins=50)
    plt.title("cosine: classifier weights vs class means (per class)")
    plt.xlabel("cosine")
    plt.ylabel("count")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc_cos_W_vs_means.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # NC2 offdiag means
    Gm = gram_cosine_matrix(means).detach().cpu().numpy()
    C = Gm.shape[0]
    target = -1.0 / (C - 1)
    off_m = Gm[~np.eye(C, dtype=bool)]
    plt.figure(figsize=(6, 4))
    plt.hist(off_m, bins=80)
    plt.axvline(target, linestyle="--")
    plt.title(f"NC2: offdiag cosine(class means), target={target:.4f}")
    plt.xlabel("cosine")
    plt.ylabel("count")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc2_offdiag_cos_means.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # NC4 offdiag weights
    Gw = gram_cosine_matrix(W).detach().cpu().numpy()
    off_w = Gw[~np.eye(C, dtype=bool)]
    plt.figure(figsize=(6, 4))
    plt.hist(off_w, bins=80)
    plt.axvline(target, linestyle="--")
    plt.title(f"NC4: offdiag cosine(classifier weights), target={target:.4f}")
    plt.xlabel("cosine")
    plt.ylabel("count")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc4_offdiag_cos_W.png")
    plt.savefig(p, dpi=200)
    plt.close()

    # NC5 offdiag hist + scaled gram + eigs
    G = nc5["gram"]
    off_vals = G[~np.eye(G.shape[0], dtype=bool)]
    target5 = nc5["target"]

    plt.figure(figsize=(6, 4))
    plt.hist(off_vals, bins=120)
    plt.axvline(target5, linestyle="--")
    plt.title(f"NC5: offdiag(Gram) distribution, target={target5:.4f}")
    plt.xlabel("off-diagonal value")
    plt.ylabel("count")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc5_offdiag_hist.png")
    plt.savefig(p, dpi=200)
    plt.close()

    absmax = float(np.quantile(np.abs(off_vals), 0.995))
    absmax = max(absmax, abs(target5) * 3.0, 1e-3)

    plt.figure(figsize=(6, 5))
    plt.imshow(G, aspect="auto", vmin=-absmax, vmax=absmax, cmap="coolwarm")
    plt.colorbar()
    plt.title(f"NC5 Gram (scaled), target offdiag={target5:.4f}")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc5_gram_scaled.png")
    plt.savefig(p, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(nc5["eigs"], bins=60)
    plt.title("NC5 Gram eigenvalues")
    plt.xlabel("eigenvalue")
    plt.ylabel("count")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "nc5_eigs.png")
    plt.savefig(p, dpi=200)
    plt.close()

    print("saved NC figs in:", FIG_DIR)


# =========================
# BONUS: NC across layers (online)
# =========================
@torch.no_grad()
def fit_linear_probe_ls_online(stream, num_classes, ridge=1e-3):
    XtX = None
    XtY = None

    for X, y_cpu in stream:
        X = X.float()
        y = y_cpu.to(device, dtype=torch.long, non_blocking=True)

        B, D = X.shape
        if XtX is None:
            XtX = torch.zeros(D + 1, D + 1, device=device, dtype=torch.float32)
            XtY = torch.zeros(D + 1, num_classes, device=device, dtype=torch.float32)

        ones = torch.ones(B, 1, device=device, dtype=torch.float32)
        Xb = torch.cat([X, ones], dim=1)

        Y = torch.zeros(B, num_classes, device=device, dtype=torch.float32)
        Y[torch.arange(B, device=device), y] = 1.0

        XtX += Xb.T @ Xb
        XtY += Xb.T @ Y

    Dt = XtX.size(0)
    XtX = XtX + ridge * torch.eye(Dt, device=device, dtype=XtX.dtype)

    Wb = torch.linalg.solve(XtX, XtY)   # [(D+1), C]
    W = Wb[:-1, :].T.contiguous()       # [C, D]
    b = Wb[-1, :].contiguous()          # [C]
    return W, b


@torch.no_grad()
def compute_class_means_online(stream, num_classes):
    sum_c = None
    count_c = torch.zeros(num_classes, device=device, dtype=torch.long)

    for X, y_cpu in stream:
        X = X.float()
        y = y_cpu.to(device, dtype=torch.long, non_blocking=True)
        _, D = X.shape
        if sum_c is None:
            sum_c = torch.zeros(num_classes, D, device=device, dtype=torch.float32)

        for c in range(num_classes):
            idx = (y == c)
            if idx.any():
                sum_c[c] += X[idx].sum(dim=0)
                count_c[c] += idx.sum()

    means = sum_c / (count_c.clamp_min(1).unsqueeze(1).float())
    return means, count_c


@torch.no_grad()
def compute_within_var_online(stream, means):
    total = 0
    acc = 0.0
    for X, y_cpu in stream:
        X = X.float()
        y = y_cpu.to(device, dtype=torch.long, non_blocking=True)
        diff = X - means[y]
        acc += diff.pow(2).sum(dim=1).sum().item()
        total += X.size(0)
    return float(acc / max(total, 1))


@torch.no_grad()
def compute_nc_all_online(layer_name, model, loader, num_classes, ridge=1e-3):
    def make_stream():
        for feats_batch, y in extract_layer_features_stream(model, loader, layer_names=(layer_name,)):
            yield feats_batch[layer_name], y

    means, counts = compute_class_means_online(make_stream(), num_classes)
    within_var = compute_within_var_online(make_stream(), means)

    global_mean = means.mean(dim=0)
    between_var = (means - global_mean).pow(2).sum(dim=1).mean().item()
    nc1 = within_var / (between_var + 1e-12)

    nc2_mean, nc2_std = nc2_metric(means)

    W_probe, b_probe = fit_linear_probe_ls_online(make_stream(), num_classes, ridge=ridge)
    nc3_mean, nc3_std, nc3_per_class = nc3_metric(W_probe, means)
    nc4_mean, nc4_std = nc4_metric(W_probe)

    return {
        "within_var": float(within_var),
        "between_var": float(between_var),
        "nc1": float(nc1),
        "nc2_mean": float(nc2_mean),
        "nc2_std": float(nc2_std),
        "nc3_mean": float(nc3_mean),
        "nc3_std": float(nc3_std),
        "nc4_mean": float(nc4_mean),
        "nc4_std": float(nc4_std),
        "nc3_per_class": nc3_per_class,
        "counts": counts.detach().cpu().numpy(),
        "probe_W": W_probe.detach().cpu(),
        "probe_b": b_probe.detach().cpu(),
    }


def save_nc_across_layers_plots(layer_results):
    names = [n for n in ["layer1", "layer2", "layer3", "layer4"] if n in layer_results]

    nc1 = [layer_results[n]["nc1"] for n in names]
    nc2 = [layer_results[n]["nc2_mean"] for n in names]
    nc3 = [layer_results[n]["nc3_mean"] for n in names]
    nc4 = [layer_results[n]["nc4_mean"] for n in names]

    plt.figure(figsize=(7, 4))
    plt.plot(nc1, marker="o")
    plt.xticks(range(len(names)), names)
    plt.title("NC1 across layers")
    plt.ylabel("within / between")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_nc1_across_layers.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(nc2, marker="o")
    plt.xticks(range(len(names)), names)
    plt.title("NC2 across layers")
    plt.ylabel("mean offdiag cosine(means)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_nc2_across_layers.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(nc3, marker="o")
    plt.xticks(range(len(names)), names)
    plt.title("NC3 across layers (linear probe LS)")
    plt.ylabel("mean cosine(Wc, meanc)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_nc3_across_layers.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(nc4, marker="o")
    plt.xticks(range(len(names)), names)
    plt.title("NC4 across layers (linear probe LS)")
    plt.ylabel("mean offdiag cosine(W)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "bonus_nc4_across_layers.png"), dpi=200)
    plt.close()


# =========================
# main
# =========================
def maybe_plot_training_curves():
    if not os.path.exists(SAVE_CURVES):
        return
    curves = torch.load(SAVE_CURVES, map_location="cpu")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(curves["train_loss"], label="train")
    plt.plot(curves["val_loss"], label="val")
    plt.title("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([100 * a for a in curves["train_acc"]], label="train")
    plt.plot([100 * a for a in curves["val_acc"]], label="val")
    plt.title("accuracy (%)")
    plt.legend()

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "training_curves.png")
    plt.savefig(p, dpi=200)
    plt.close()
    print("saved fig:", p)


def main():
    print("device:", device, "| amp:", USE_AMP)
    print("data_dir:", DATA_DIR)
    print("out_dir:", OUT_DIR)

    train_loader_eval, _, test_loader, mean, std = build_cifar100_loaders()
    ood_loader = build_ood_loader(mean, std)

    model = ResNet18(num_classes=NUM_CLASSES).to(device)
    load_checkpoint(SAVE_PATH_BEST, model, map_location=device)

    if not os.path.exists(SAVE_TRAIN_FEATS):
        raise FileNotFoundError(f"missing: {SAVE_TRAIN_FEATS}")
    saved_feats = torch.load(SAVE_TRAIN_FEATS, map_location="cpu")
    tr_feats = saved_feats["train_feats"].float()
    tr_y = saved_feats["train_labels"].long()

    if not os.path.exists(SAVE_NECO_PARAMS):
        raise FileNotFoundError(f"missing: {SAVE_NECO_PARAMS}")
    neco_params = torch.load(SAVE_NECO_PARAMS, map_location="cpu")

    maybe_plot_training_curves()

    # ---- OOD
    if RUN_OOD_SCORES:
        print("\n== extracting ID/OOD logits+features ==")
        id_logits, id_feats, _ = extract_logits_features(model, test_loader)
        ood_logits, ood_feats, _ = extract_logits_features(model, ood_loader)

        id_msp, ood_msp = score_msp(id_logits).numpy(), score_msp(ood_logits).numpy()
        id_ml, ood_ml = score_maxlogit(id_logits).numpy(), score_maxlogit(ood_logits).numpy()
        id_en, ood_en = score_energy(id_logits).numpy(), score_energy(ood_logits).numpy()

        class_means, precision = fit_mahalanobis(tr_feats, tr_y, num_classes=NUM_CLASSES)
        id_ma = score_mahalanobis(id_feats, class_means, precision).numpy()
        ood_ma = score_mahalanobis(ood_feats, class_means, precision).numpy()

        W_fc = model.fc.weight.detach().cpu()
        vim_params = fit_vim(tr_feats, W_fc, r_dim=64)
        id_vm = score_vim(id_logits, id_feats, vim_params).numpy()
        ood_vm = score_vim(ood_logits, ood_feats, vim_params).numpy()

        id_ne = score_neco(id_logits, id_feats, neco_params).numpy()
        ood_ne = score_neco(ood_logits, ood_feats, neco_params).numpy()

        print("\n== OOD metrics (higher score = more ID) ==")
        res = {
            "MSP": summarize_method("MSP", id_msp, ood_msp),
            "MaxLogit": summarize_method("MaxLogit", id_ml, ood_ml),
            "Energy": summarize_method("Energy", id_en, ood_en),
            "Mahalanobis": summarize_method("Mahalanobis", id_ma, ood_ma),
            "ViM": summarize_method("ViM", id_vm, ood_vm),
            "NECO": summarize_method("NECO", id_ne, ood_ne),
        }

        save_hist(id_msp, ood_msp, f"MSP (ID=CIFAR100 test vs OOD={OOD_NAME})", "hist_msp.png")
        save_hist(id_ml, ood_ml, f"MaxLogit (ID=CIFAR100 test vs OOD={OOD_NAME})", "hist_maxlogit.png")
        save_hist(id_en, ood_en, f"Energy (ID=CIFAR100 test vs OOD={OOD_NAME})", "hist_energy.png")
        save_hist(id_ma, ood_ma, f"Mahalanobis (ID=CIFAR100 test vs OOD={OOD_NAME})", "hist_mahalanobis.png")
        save_hist(id_vm, ood_vm, f"ViM (ID=CIFAR100 test vs OOD={OOD_NAME})", "hist_vim.png")
        save_hist(id_ne, ood_ne, f"NECO (ID=CIFAR100 test vs OOD={OOD_NAME})", "hist_neco.png")

        out_path = os.path.join(OUT_DIR, f"ood_results_{OOD_NAME}.pt")
        torch.save({"ood_name": OOD_NAME, "results": res}, out_path)
        print("saved:", out_path)

    # ---- NC1..NC5 (penultimate)
    if RUN_NC_1_TO_5:
        print("\n== Neural Collapse (NC1..NC5) on train penultimate feats ==")
        means, within_var, counts = compute_class_stats(tr_feats, tr_y, NUM_CLASSES)

        W = model.fc.weight.detach().cpu()
        nc1, between_var = nc1_metric(within_var, means)
        nc2_mean, nc2_std = nc2_metric(means)
        nc3_mean, nc3_std, nc3_per_class = nc3_metric(W, means)
        nc4_mean, nc4_std = nc4_metric(W)

        offdiag_dists = pairwise_mean_distances(means)
        nc5 = nc5_metric(means)

        print("NC1:", nc1)
        print("NC2 mean/std:", nc2_mean, nc2_std)
        print("NC3 mean/std:", nc3_mean, nc3_std)
        print("NC4 mean/std:", nc4_mean, nc4_std)
        print("NC5 target:", nc5["target"])
        print("NC5 abs dev mean:", nc5["abs_dev_mean"])
        print("NC5 fro dev:", nc5["fro_dev"])

        save_nc_plots(offdiag_dists, within_var, between_var, nc3_per_class, nc5, means, W)

        out_nc = {
            "within_class_variance": within_var,
            "between_class_variance": between_var,
            "nc1_ratio_within_over_between": nc1,
            "nc2_offdiag_cos_mean_means": nc2_mean,
            "nc2_offdiag_cos_std_means": nc2_std,
            "nc3_cos_mean_W_vs_means": nc3_mean,
            "nc3_cos_std_W_vs_means": nc3_std,
            "nc3_per_class_cos": nc3_per_class,
            "nc4_offdiag_cos_mean_W": nc4_mean,
            "nc4_offdiag_cos_std_W": nc4_std,
            "class_mean_dist_offdiag": offdiag_dists,
            "class_counts": counts.numpy(),
            "nc5": nc5,
        }
        out_path = os.path.join(OUT_DIR, "nc_1_to_5_penultimate_train.pt")
        torch.save(out_nc, out_path)
        print("saved:", out_path)

    # ---- BONUS: NC across layers
    if RUN_NC_ACROSS_LAYERS:
        print("\n== BONUS: NC across layers (layer1..layer4) ==")
        layer_results = {}
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            layer_results[lname] = compute_nc_all_online(lname, model, train_loader_eval, NUM_CLASSES, ridge=1e-3)
            print(
                lname,
                "| NC1", layer_results[lname]["nc1"],
                "| NC2", layer_results[lname]["nc2_mean"],
                "| NC3", layer_results[lname]["nc3_mean"],
                "| NC4", layer_results[lname]["nc4_mean"],
            )

        save_nc_across_layers_plots(layer_results)

        out_path = os.path.join(OUT_DIR, "bonus_nc_across_layers.pt")
        torch.save(layer_results, out_path)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
