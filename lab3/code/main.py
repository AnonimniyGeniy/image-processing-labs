"""ЛР3: небольшая CNN для классификации рукописных цифр MNIST."""
import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import environment, read_image, save_json
from .digits import download_mnist, load_own_digits, normalize, preprocess_digit, read_idx, stratified_split


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32*7*7, 64), nn.ReLU(),
                                        nn.Dropout(.25), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def choose_device(value):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classification_metrics(truth, predicted):
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(10), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros(10), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(10), where=precision+recall != 0)
    return {"size": int(matrix.sum()), "accuracy": float(tp.sum()/matrix.sum()),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
            "per_class": [{"digit": i, "support": int(matrix[i].sum()),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i in range(10)]}


def make_loader(images, labels, batch_size=256, shuffle=False, seed=42):
    dataset = TensorDataset(normalize(images), torch.from_numpy(labels.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truths, predictions, probabilities = [], [], []
    total_loss = 0.0
    for images, labels in loader:
        logits = model(images.to(device))
        total_loss += nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item()
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        truths.append(labels.numpy())
    truth, predicted = np.concatenate(truths), np.concatenate(predictions)
    metrics = classification_metrics(truth, predicted)
    metrics["loss"] = total_loss / len(truth)
    return metrics, predicted, np.concatenate(probabilities)


def plot_results(output, images, truth, predicted, metrics, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = np.array(metrics["confusion_matrix"])
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
           title=f"{prefix}: confusion matrix ({len(truth)} images)")
    for y in range(10):
        for x in range(10):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[y, x] > matrix.max()/2 else "black")
    fig.colorbar(im, ax=ax, label="Image count")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_confusion.png", dpi=150)
    plt.close(fig)
    wrong = np.flatnonzero(truth != predicted)
    selected = wrong[:24] if len(wrong) else np.arange(min(24, len(images)))
    columns = min(6, len(selected))
    rows = (len(selected)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(max(5, columns*1.7), rows*1.8+0.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, selected):
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true {truth[index]}, pred {predicted[index]}", fontsize=9)
    fig.suptitle(f"{prefix}: " + ("first classification errors" if len(wrong) else "sample predictions"))
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_examples.png", dpi=150)
    plt.close(fig)


def train(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs и batch_size должны быть >= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    download_mnist(args.data)
    images = read_idx(args.data / "train-images-idx3-ubyte.gz")
    labels = read_idx(args.data / "train-labels-idx1-ubyte.gz")
    train_indices, val_indices = stratified_split(labels, 5000, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "split_indices.npz", train=train_indices, validation=val_indices)
    train_loader = make_loader(images[train_indices], labels[train_indices], args.batch_size, True, args.seed)
    train_eval_loader = make_loader(images[train_indices], labels[train_indices], args.batch_size)
    val_loader = make_loader(images[val_indices], labels[val_indices], args.batch_size)
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_epoch, history = float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        tick = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
        validation, _, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_online_loss": total_loss/count,
               "train_online_accuracy": correct/count, "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
               "elapsed_s": time.perf_counter()-tick}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}: train loss={row['train_online_loss']:.4f}; "
              f"val loss={row['val_loss']:.4f}; val acc={row['val_accuracy']:.4%}; "
              f"{row['elapsed_s']:.1f} s", flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "architecture": "DigitCNN-v1", "epoch": epoch,
                        "seed": args.seed, "normalization": {"mean": .1307, "std": .3081}},
                       args.output / "best_model.pt")
        save_json(args.output / "history.json", history)
    checkpoint = torch.load(args.output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    final_train, _, _ = evaluate(model, train_eval_loader, device)
    final_val, _, _ = evaluate(model, val_loader, device)
    # Официальный test загружается для оценки только после фиксации лучшей эпохи.
    test_images = read_idx(args.data / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx(args.data / "t10k-labels-idx1-ubyte.gz")
    official_test, predicted, _ = evaluate(model, make_loader(test_images, test_labels), device)
    report = {"environment": {**environment(), "torch": torch.__version__, "device": str(device)},
              "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
              "optimizer": "Adam(lr=0.001, weight_decay=0.0001)", "loss": "CrossEntropyLoss",
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "selection": "minimum validation cross-entropy", "best_epoch": best_epoch,
              "training_elapsed_s": time.perf_counter()-start,
              "train_eval": final_train, "validation": final_val, "official_mnist_test": official_test,
              "own_test_status": "not_collected; official MNIST test is not a substitute",
              "determinism": "Fixed seeds/split; exact MPS/CUDA bitwise reproducibility is not guaranteed"}
    save_json(args.output / "metrics.json", report)
    plot_results(args.output, test_images, test_labels, predicted, official_test, "official_test")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylabel in zip(axes, ["loss", "accuracy"], ["Cross-entropy (unitless)", "Accuracy (fraction)"]):
        ax.plot([r["epoch"] for r in history], [r[f"train_online_{metric}"] for r in history], "o-", label="train (online)")
        ax.plot([r["epoch"] for r in history], [r[f"val_{metric}"] for r in history], "o-", label="validation (eval)")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"MNIST: {metric}")
        ax.axvline(best_epoch, linestyle="--", color="gray", linewidth=1, label="selected epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", dpi=150)
    plt.close(fig)
    print(f"Official MNIST test accuracy: {official_test['accuracy']:.4%}. "
          "Самостоятельный тест ещё не собран.", flush=True)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "DigitCNN-v1":
        raise ValueError("Неподдерживаемая архитектура checkpoint")
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_own(args):
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    images, labels, names = load_own_digits(args.folder, args.json)
    metrics, predicted, probabilities = evaluate(model, make_loader(images, labels), device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source": "user-collected handwriting; provenance is declared by the collector",
              "checkpoint": str(args.checkpoint), "metrics": metrics,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "samples_per_class": np.bincount(labels, minlength=10).tolist(),
              "source_json_sha256": hashlib.sha256(args.json.read_bytes()).hexdigest() if args.json else None,
              "preprocessing": "auto polarity, Otsu foreground, fit 20x20, center of mass, normalize",
              "warning": "Keep this set independent; do not tune model on its results"}
    save_json(args.output / "own_test_metrics.json", report)
    with (args.output / "own_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "true", "predicted", "softmax_score"])
        for name, truth, pred, probs in zip(names, labels, predicted, probabilities):
            writer.writerow([name, int(truth), int(pred), float(probs[pred])])
    plot_results(args.output, images, labels, predicted, metrics, "own_test")
    print(f"Собственный тест: {len(images)} изображений, accuracy={metrics['accuracy']:.2%}")


def predict(args):
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    image = preprocess_digit(read_image(args.input, cv2.IMREAD_UNCHANGED))
    with torch.no_grad():
        probabilities = model(normalize(image[None]).to(device)).softmax(dim=1)[0].cpu().numpy()
    digit = int(probabilities.argmax())
    print(f"Цифра: {digit}; softmax score: {probabilities[digit]:.4f} (не калиброванная вероятность)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("lab3/data/mnist"))
    training.add_argument("--output", type=Path, default=Path("lab3/results"))
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    training.set_defaults(func=train)
    own = commands.add_parser("evaluate-own")
    source = own.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", type=Path)
    source.add_argument("--json", type=Path)
    own.add_argument("--output", type=Path, default=Path("lab3/results"))
    prediction = commands.add_parser("predict")
    prediction.add_argument("--input", type=Path, required=True)
    for sub in [own, prediction]:
        sub.add_argument("--checkpoint", type=Path, default=Path("lab3/results/best_model.pt"))
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    own.set_defaults(func=evaluate_own)
    prediction.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
