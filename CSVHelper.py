import csv
import os

def init_metric_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run",
                "model",
                "split",
                "epoch",
                "overall_acc",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "lr",
                "train_loss",
            ])


def append_metric_csv(
    csv_path,
    run,
    model,
    split,
    epoch,
    overall_acc,
    macro_precision,
    macro_recall,
    macro_f1,
    lr=None,
    train_loss=None,
):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            model,
            split,
            epoch,
            overall_acc,
            macro_precision,
            macro_recall,
            macro_f1,
            "" if lr is None else lr,
            "" if train_loss is None else train_loss,
        ])