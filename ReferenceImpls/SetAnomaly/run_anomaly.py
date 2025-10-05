import torch, platform, csv, os, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

import numpy as np
import models
import AnomalySetGenerator
from torch.utils.data import Dataset, DataLoader

d = 2 # Individual Set Item Channels
N = 64 # Set Size
DS = 4096 # Dataset Size
batch_size = 128 # Batch Size
learning_rate = 1e-4
num_epochs = 1000
num_repeats = 1
class CSVLogger:
    """Append metrics as rows to a CSV that Excel can open directly."""
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._ensure_header()

    def _ensure_header(self):
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if write_header:
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    def append(self, row: dict):
        # Only keep known fields to avoid accidental extras
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean)

    def reset_path(self,path):
        self.path = path
        self._ensure_header()


class SetTrainer(object):
    def __init__(self):
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch built with CUDA:", torch.version.cuda)
        print("OS:", platform.platform())
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PCSNetwork = models.PermutationEquivariantModel(N,d,d).to(self.device)
        self.DeepSetsNet = models.DeepSetsModel(d,d).to(self.device)
        self.optimizer = optim.AdamW([{'params': self.PCSNetwork.parameters()}], lr=learning_rate,weight_decay=1e-5, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(100,num_epochs,100)),gamma=0.1)
        self.logger = CSVLogger(
            "PCS_SetAnomaly_LOG.csv",
            fieldnames=[
                "timestamp", "epoch",
                "loss", "running_train_acc","anomaly_train_acc", "lr"
            ]
        )
        print("Setting up Training Dataset")
        self.data = AnomalySetGenerator.AnomalySetDataset(n_samples=DS, set_size=N, d=d, p_anom=0.5, base_sigma=1.0)
        print("Setting up Validation Dataset")
        self.val = AnomalySetGenerator.AnomalySetDataset(
        n_samples=int(DS/4), set_size=N, d=d, p_anom=0.5, seed=7
    )
        torch.backends.cudnn.benchmark = True

    def train_PCS(self):
        self.PCSNetwork.train()
        self.PCSNetwork.to(self.device)

        self.optimizer = optim.AdamW(self.PCSNetwork.parameters(),
                                     lr=learning_rate, weight_decay=1e-4, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=list(range(400, num_epochs, 400)), gamma=0.1
        )
        self.logger.reset_path("PCS_SetAnomaly_LOG.csv")

        dl = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        for epoch in trange(num_epochs, desc="Epochs: "):
            running_loss = 0.0
            total_elems = 0
            tp = tn = fp = fn = 0

            for x, y in tqdm(dl, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                x = x.to(self.device, non_blocking=True)  # (B,N,C)
                y = y.to(self.device, non_blocking=True)  # (B,N) 0/1

                self.optimizer.zero_grad(set_to_none=True)

                # per-set weights (no grad) with clamping
                with torch.no_grad():
                    pos_per = y.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
                    neg_per = (y.size(1) - pos_per).clamp(min=1)
                    w_pos = (neg_per / pos_per)  # (B,1)
                    weight = torch.ones_like(y, dtype=torch.float)
                    weight[y == 1] = w_pos.expand_as(y)[y == 1]

                logits = self.PCSNetwork(x)  # (B,N)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y.float(), weight=weight, reduction="mean"
                )
                running_loss += loss.item() * y.numel()
                total_elems += y.numel()

                # metrics (no grad)
                with torch.no_grad():
                    preds = (logits.sigmoid() >= 0.5)
                    tp += ((preds == 1) & (y == 1)).sum().item()
                    tn += ((preds == 0) & (y == 0)).sum().item()
                    fp += ((preds == 1) & (y == 0)).sum().item()
                    fn += ((preds == 0) & (y == 1)).sum().item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.PCSNetwork.parameters(), 1.0)
                self.optimizer.step()

            # epoch metrics
            epoch_loss = running_loss / max(total_elems, 1)
            overall_acc = (tp + tn) / max(total_elems, 1)
            tpr = tp / max(tp + fn, 1)  # anomaly recall
            tnr = tn / max(tn + fp, 1)  # normal specificity
            balanced_acc = 0.5 * (tpr + tnr)

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                self.logger.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "running_train_acc": overall_acc,
                    "anomaly_train_acc": tpr,
                    "lr": self.optimizer.param_groups[0]['lr'],
                })
                tqdm.write(
                    f"After epoch {epoch + 1}  Acc: {overall_acc:.3f}  AnomRecall: {tpr:.3f}  BalAcc: {balanced_acc:.3f}")

    def train_DeepSets(self):
        self.DeepSetsNet.train()
        self.DeepSetsNet.to(self.device)

        self.optimizer = optim.AdamW(self.DeepSetsNet.parameters(),
                                     lr=learning_rate, weight_decay=1e-4, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=list(range(400, num_epochs, 400)), gamma=0.1
        )
        self.logger.reset_path("DeepSets_SetAnomaly_LOG.csv")

        dl = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        for epoch in trange(num_epochs, desc="Epochs: "):
            running_loss = 0.0
            total_elems = 0
            tp = tn = fp = fn = 0

            for x, y in tqdm(dl, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                x = x.to(self.device, non_blocking=True)  # (B,N,C)
                y = y.to(self.device, non_blocking=True)  # (B,N) 0/1

                self.optimizer.zero_grad(set_to_none=True)

                # per-set weights (no grad) with clamping
                with torch.no_grad():
                    pos_per = y.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
                    neg_per = (y.size(1) - pos_per).clamp(min=1)
                    w_pos = (neg_per / pos_per)  # (B,1)
                    weight = torch.ones_like(y, dtype=torch.float)
                    weight[y == 1] = w_pos.expand_as(y)[y == 1]

                logits = self.DeepSetsNet(x)  # (B,N)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y.float(), weight=weight, reduction="mean"
                )
                running_loss += loss.item() * y.numel()
                total_elems += y.numel()

                # metrics (no grad)
                with torch.no_grad():
                    preds = (logits.sigmoid() >= 0.5)
                    tp += ((preds == 1) & (y == 1)).sum().item()
                    tn += ((preds == 0) & (y == 0)).sum().item()
                    fp += ((preds == 1) & (y == 0)).sum().item()
                    fn += ((preds == 0) & (y == 1)).sum().item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.DeepSetsNet.parameters(), 1.0)
                self.optimizer.step()

            # epoch metrics
            epoch_loss = running_loss / max(total_elems, 1)
            overall_acc = (tp + tn) / max(total_elems, 1)
            tpr = tp / max(tp + fn, 1)  # anomaly recall
            tnr = tn / max(tn + fp, 1)  # normal specificity
            balanced_acc = 0.5 * (tpr + tnr)

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                self.logger.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "running_train_acc": overall_acc,
                    "anomaly_train_acc": tpr,
                    "lr": self.optimizer.param_groups[0]['lr'],
                })
                tqdm.write(
                    f"After epoch {epoch + 1}  Acc: {overall_acc:.3f}  AnomRecall: {tpr:.3f}  BalAcc: {balanced_acc:.3f}")

    def test_PCS(self):
        self.PCSNetwork.eval()
        dl = DataLoader(self.val, batch_size=batch_size, shuffle=False,
                        pin_memory=(self.device.type == "cuda"))

        total = tp = tn = fp = fn = 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.PCSNetwork(x)
                probs = logits.sigmoid()
                preds = (probs >= 0.5)

                tp += ((preds == 1) & (y == 1)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()
                total += y.numel()

        overall_acc = (tp + tn) / max(total, 1)
        tpr = tp / max(tp + fn, 1)  # anomaly recall
        tnr = tn / max(tn + fp, 1)  # specificity
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-8)
        balanced = 0.5 * (tpr + tnr)

        print(f"Overall per-point accuracy: {overall_acc:.3f}")
        print(f"Anomaly recall (TPR):       {tpr:.3f}")
        print(f"Precision:                  {precision:.3f}")
        print(f"F1 score:                   {f1:.3f}")
        print(f"Balanced accuracy:          {balanced:.3f}")

        return overall_acc

    def test_DeepSets(self):
        self.DeepSetsNet.eval()
        dl = DataLoader(self.val, batch_size=batch_size, shuffle=False,
                        pin_memory=(self.device.type == "cuda"))

        total = tp = tn = fp = fn = 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.DeepSetsNet(x)
                probs = logits.sigmoid()
                preds = (probs >= 0.5)

                tp += ((preds == 1) & (y == 1)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()
                total += y.numel()

        overall_acc = (tp + tn) / max(total, 1)
        tpr = tp / max(tp + fn, 1)  # anomaly recall
        tnr = tn / max(tn + fp, 1)  # specificity
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-8)
        balanced = 0.5 * (tpr + tnr)

        print(f"Overall per-point accuracy: {overall_acc:.3f}")
        print(f"Anomaly recall (TPR):       {tpr:.3f}")
        print(f"Precision:                  {precision:.3f}")
        print(f"F1 score:                   {f1:.3f}")
        print(f"Balanced accuracy:          {balanced:.3f}")

        return overall_acc

if __name__ == "__main__":
    PCS_test_accs = []
    DeepSets_test_accs = []
    for i in range(num_repeats):
        print('='*30 + ' Start Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
        t = SetTrainer()
        t.train_PCS()
        acc = t.test_PCS()
        PCS_test_accs.append(acc)
        t.train_DeepSets()
        acc = t.test_DeepSets()
        DeepSets_test_accs.append(acc)
        print('=' * 30 + ' Finish Run {0}/{1} '.format(i + 1, num_repeats) + '=' * 30)
    print('\n')
    if num_repeats > 2:
        try:
            mean_acc = np.mean(PCS_test_accs)
            std_acc = np.std(PCS_test_accs)
            mean_acc2 = np.mean(DeepSets_test_accs)
            std_acc2 = np.std(DeepSets_test_accs)
            print(f"PCS Test accuracy: {mean_acc: 0.2f} ± {std_acc:0.3f}")
            print(f"DeepSets Test accuracy: {mean_acc2: 0.2f} ± {std_acc2:0.3f}")
        except:
            print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(PCS_test_accs), np.std(PCS_test_accs)))
            print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(DeepSets_test_accs), np.std(DeepSets_test_accs)))





