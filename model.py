import os
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms as T
import numpy as np

SEED = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

PATH = DATA_DIR

SAVE_PATH_BEST = os.path.join(OUT_DIR, "resnet18_cifar100_best.pt")
SAVE_PATH_LAST = os.path.join(OUT_DIR, "resnet18_cifar100_last.pt")
SAVE_SPLITS = os.path.join(OUT_DIR, "cifar100_splits.pt")
SAVE_TRAIN_FEATS = os.path.join(OUT_DIR, "train_penultimate_feats_labels.pt")
SAVE_NECO_PARAMS = os.path.join(OUT_DIR, "neco_params.pt")
SAVE_CLASS_STATS = os.path.join(OUT_DIR, "class_stats_train.pt")
SAVE_HEAD = os.path.join(OUT_DIR, "classifier_head.pt")
SAVE_CURVES = os.path.join(OUT_DIR, "training_curves.pt")

slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus is not None:
    NUM_WORKERS = max(0, int(slurm_cpus) - 1)
else:
    NUM_WORKERS = min(8, os.cpu_count() or 4)

VAL_SIZE = 5000
TEST_SIZE = 5000
MINIBATCH_SIZE = 512
EPOCHS = 300
NUM_CLASSES = 100
NECO_D = 64

PREFETCH_FACTOR = 4
PIN = (device.type == "cuda")
USE_AMP = (device.type == "cuda")
USE_COMPILE = False

mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]

transform_cifar100_train = T.Compose([
    T.RandomHorizontalFlip(p=0.3),
    T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05),
    T.RandomApply([T.RandomRotation(10), T.Resize(40), T.CenterCrop(32)], p=0.1),
    T.ToTensor(),
    T.Normalize(mean, std),
])

transform_cifar100_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_ds = datasets.CIFAR100(PATH, train=True, download=True, transform=transform_cifar100_train)
train_ds_eval = datasets.CIFAR100(PATH, train=True, download=False, transform=transform_cifar100_test)
test_ds_full = datasets.CIFAR100(PATH, train=False, download=True, transform=transform_cifar100_test)

g = torch.Generator().manual_seed(SEED)
val_ds, test_ds = random_split(test_ds_full, [VAL_SIZE, TEST_SIZE], generator=g)

torch.save(
    {
        "val_indices": val_ds.indices,
        "test_indices": test_ds.indices,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "mean": mean,
        "std": std,
    },
    SAVE_SPLITS
)
print("saved:", SAVE_SPLITS)

dl_kwargs = dict(
    batch_size=MINIBATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
)
if NUM_WORKERS > 0:
    dl_kwargs["prefetch_factor"] = PREFETCH_FACTOR

train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)
test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **dl_kwargs)
train_loader_eval = DataLoader(train_ds_eval, shuffle=False, drop_last=False, **dl_kwargs)

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
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
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

model = ResNet18(num_classes=NUM_CLASSES).to(device)

if USE_COMPILE and device.type == "cuda" and hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", str(e))

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, best_val_acc=None):
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "mean": mean,
        "std": std,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, path)
    print("saved:", path)

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total

def train(model, optimizer, scheduler=None, epochs=50):
    best_val_acc = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    curves = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            loss_sum += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = accuracy(model, val_loader)

        lr_now = optimizer.param_groups[0]["lr"]
        curves["train_loss"].append(train_loss)
        curves["train_acc"].append(train_acc)
        curves["val_loss"].append(val_loss)
        curves["val_acc"].append(val_acc)
        curves["lr"].append(lr_now)

        print(
            f"epoch {epoch+1:03d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {lr_now:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                SAVE_PATH_BEST, model,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch+1, best_val_acc=best_val_acc
            )

    save_checkpoint(
        SAVE_PATH_LAST, model,
        optimizer=optimizer, scheduler=scheduler,
        epoch=epochs, best_val_acc=best_val_acc
    )

    torch.save(curves, SAVE_CURVES)
    print("saved:", SAVE_CURVES)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.95, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-1,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.43,
    div_factor=10,
    final_div_factor=1000,
    three_phase=True,
)

@torch.no_grad()
def extract_logits_features(model, loader):
    model.eval()

    all_logits, all_feats, all_labels = [], [], []
    feats_buf = {}

    def hook_fc_input(module, inp, out):
        feats_buf["feats"] = inp[0].detach()

    h = model.fc.register_forward_hook(hook_fc_input)

    for x, y in loader:
        feats_buf.clear()
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(x)

        if "feats" not in feats_buf:
            raise RuntimeError("hook did not capture feats; check model.fc hook")
        feats = feats_buf["feats"]

        all_logits.append(logits.detach().cpu())
        all_feats.append(feats.detach().cpu())
        all_labels.append(y.detach().cpu())

    h.remove()

    logits = torch.cat(all_logits, dim=0)
    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, feats, labels

def fit_neco_pca(train_feats, d=64):
    X = train_feats.float()
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    q = min(d, Xc.shape[1], Xc.shape[0] - 1)
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False)
    Vd = V[:, :q].contiguous()
    return mu.squeeze(0), Vd

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

if __name__ == "__main__":
    print("device:", device)
    print("num_workers:", NUM_WORKERS, "| pin_memory:", PIN, "| amp:", USE_AMP)
    print("data_dir:", DATA_DIR)
    print("out_dir:", OUT_DIR)

    train(model, optimizer, scheduler=scheduler, epochs=EPOCHS)

    test_loss, test_acc = accuracy(model, test_loader)
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.4f}")

    W = model.fc.weight.detach().cpu()
    b = model.fc.bias.detach().cpu() if model.fc.bias is not None else None
    torch.save({"W": W, "b": b}, SAVE_HEAD)
    print("saved:", SAVE_HEAD)

    _, tr_feats, tr_y = extract_logits_features(model, train_loader_eval)

    tr_feats_to_save = tr_feats.half()
    torch.save(
        {
            "train_feats": tr_feats_to_save,
            "train_labels": tr_y,
            "seed": SEED,
            "neco_d": NECO_D,
        },
        SAVE_TRAIN_FEATS
    )
    print("saved:", SAVE_TRAIN_FEATS)

    mu, Vd = fit_neco_pca(tr_feats, d=NECO_D)
    torch.save({"mu": mu, "Vd": Vd, "d": NECO_D, "seed": SEED}, SAVE_NECO_PARAMS)
    print("saved:", SAVE_NECO_PARAMS)

    means, within_var, counts = compute_class_stats(tr_feats, tr_y, NUM_CLASSES)
    torch.save(
        {
            "class_means": means,
            "within_var": within_var,
            "class_counts": counts,
            "seed": SEED,
        },
        SAVE_CLASS_STATS
    )
    print("saved:", SAVE_CLASS_STATS)
