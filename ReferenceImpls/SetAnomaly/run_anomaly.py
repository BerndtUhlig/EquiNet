import torch, platform, csv, os, time
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import pandas as pd
import BanKData
import numpy as np
import models
import AnomalySetGenerator
from torch.utils.data import Dataset, DataLoader, Subset
from MSL.MSLDataset import MSLDataset
import random
import torch
from CSVHelper import append_metric_csv,init_metric_csv

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dimension = 2 # Individual Set Item Channels
N = 512 # Set Size
DS = 4096 # Dataset Size
batch_size = 128 # Batch Size
learning_rate = 1e-4
num_epochs = 1000
num_repeats = 3
key = "SetAnomaly"
csv_paths = {
    #"MSL":"",
    "SetAnomaly":"",
    #"Bank":"bank-additional-full_normalised.csv",
}
filenamePCS = f"./ReferenceImpls/SetAnomaly/results/test_accs_{time.strftime('%Y-%m-%d-%H-%M-%S')}.txt"
metrics_PCS_csv = f"./ReferenceImpls/SetAnomaly/results/val_metrics_PCS_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"
metrics_DEEPSETS_csv = f"./ReferenceImpls/SetAnomaly/results/val_metrics_DEEPSETS_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"

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
    def __init__(self, key:str = ""):
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch built with CUDA:", torch.version.cuda)
        print("OS:", platform.platform())
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.key = key
        if "Bank" in key:
            df = pd.read_csv(csv_paths[key])
            X = df.iloc[:, :-1].values.astype("float32")
            d = len(X[0])
            y = df.iloc[:, -1].values.astype("int64")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                              stratify=y, random_state=42)

            self.data = BanKData.SetFromInstances(X_train, y_train, set_size=N, n_sets=DS)
            self.val = BanKData.SetFromInstances(X_val, y_val, set_size=N, n_sets=DS)
            self.PCSNetwork = models.PermutationEquivariantModelBank(N, d, d).to(self.device)
            self.DeepSetsNet = models.DeepSetsModelBank(d, d).to(self.device)

        elif "MSL" in key:
            self.data = MSLDataset("E-13", window=N, stride=16, split="train")
            self.val = MSLDataset("E-13", window=N, stride=16, split="val")
            self.test = MSLDataset("E-13", window=N, stride=16, split="test")
            _, _, d = self.data.X.shape
            self.PCSNetwork = models.PermutationEquivariantModel(N, d, d).to(self.device)
            self.DeepSetsNet = models.DeepSetsModel(d, d).to(self.device)
        else:
            d = dimension
            print("Setting up Training Dataset")
            self.data = AnomalySetGenerator.FixedAnomalySetDataset(n_samples=DS, set_size=N, d=d, p_anom=0.5, base_sigma=1.0)
            print("Setting up Validation Dataset")
            self.val = AnomalySetGenerator.FixedAnomalySetDataset(
            n_samples=int(DS/4), set_size=N, d=d, p_anom=0.5, seed=7)
            self.test = AnomalySetGenerator.FixedAnomalySetDataset(
            n_samples=int(DS/4), set_size=N, d=d, p_anom=0.5, seed=12)
            self.PCSNetwork = models.PermutationEquivariantModel(N, d, d).to(self.device)
            self.DeepSetsNet = models.DeepSetsModel(d, d).to(self.device)


        self.PCS_log_path = f"./ReferenceImpls/SetAnomaly/results/{key}_PCSLOG.csv"
        self.DeepSets_log_path =f"./ReferenceImpls/SetAnomaly/results/{key}_DeepSetsLOG.csv"
        self.optimizer = optim.AdamW([{'params': self.PCSNetwork.parameters()}], lr=learning_rate, weight_decay=1e-5,
                                     eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(100, num_epochs, 100)),
                                                        gamma=0.1)
        self.logger = CSVLogger(
            self.PCS_log_path,
            fieldnames=[
                "timestamp", "epoch",
                "loss", "running_train_acc", "anomaly_train_acc", "lr"
            ]
        )


    def train_PCS(self):
        init_metric_csv(metrics_PCS_csv)
        self.PCSNetwork.train()
        self.PCSNetwork.to(self.device)

        self.optimizer = optim.AdamW(self.PCSNetwork.parameters(),
                                     lr=learning_rate, weight_decay=1e-4, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=list(range(400, num_epochs, 400)), gamma=0.1
        )
        self.logger.reset_path(self.PCS_log_path)

        dl = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        best_val_f1 = -1.0
        best_state = None
        best_epoch = -1

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

            if epoch % 20 == 0:
                val_metrics = self.evaluate_model("PCS",self.PCSNetwork, self.val)
                val_f1 = val_metrics[3]

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in
                                  self.PCSNetwork.state_dict().items()}
                    best_epoch = epoch + 1


            # epoch metrics
            epoch_loss = running_loss / max(total_elems, 1)
            overall_acc = (tp + tn) / max(total_elems, 1)
            tpr = tp / max(tp + fn, 1)  # anomaly recall
            tnr = tn / max(tn + fp, 1)  # normal specificity
            balanced_acc = 0.5 * (tpr + tnr)

            self.scheduler.step()

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
        if best_state is not None:
            with open(filenamePCS, "a", encoding='utf-8') as f:
                f.write(f"BEST VALIDATION AT EPOCH: {best_epoch}\n")
            self.PCSNetwork.load_state_dict(best_state)



    def train_DeepSets(self):
        init_metric_csv(metrics_DEEPSETS_csv)
        self.DeepSetsNet.train()
        self.DeepSetsNet.to(self.device)

        self.optimizer = optim.AdamW(self.DeepSetsNet.parameters(),
                                     lr=learning_rate, weight_decay=1e-4, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=list(range(400, num_epochs, 400)), gamma=0.1
        )
        self.logger.reset_path(self.DeepSets_log_path)

        dl = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        best_val_f1 = -1.0
        best_state = None
        best_epoch = -1

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

            if epoch % 20 == 0:
                val_metrics = self.evaluate_model("DEEPSETS", self.DeepSetsNet, self.val)
                val_f1 = val_metrics[3]

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in self.DeepSetsNet.state_dict().items()}
                    best_epoch = epoch + 1


            # epoch metrics
            epoch_loss = running_loss / max(total_elems, 1)
            overall_acc = (tp + tn) / max(total_elems, 1)
            tpr = tp / max(tp + fn, 1)  # anomaly recall
            tnr = tn / max(tn + fp, 1)  # normal specificity
            balanced_acc = 0.5 * (tpr + tnr)

            self.scheduler.step()



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

        if best_state is not None:
            with open(filenamePCS, "a", encoding='utf-8') as f:
                f.write(f"BEST VALIDATION AT EPOCH: {best_epoch}\n")
            self.DeepSetsNet.load_state_dict(best_state)

    def log_test(self, item:str, data:dict):
        testpath = self.key + "_TEST"+item+".csv"
        write_header = not os.path.exists(testpath) or os.path.getsize(testpath) == 0
        if write_header:
            with open(testpath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writeheader()
        clean = {k: data.get(k, "") for k in data.keys()}
        with open(testpath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(clean)

    def test_PCS(self):
        return self.evaluate_model("PCS", self.PCSNetwork, self.test, print_vals=True)

    def test_DeepSets(self):
        return self.evaluate_model("DEEPSETS", self.DeepSetsNet,self.test,print_vals=True)


    def evaluate_model(self,name, model, dataset, num_classes=13, print_vals=False):
        model.eval()
        total = tp = tn = fp = fn = 0

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

        with torch.no_grad():
            for x, y in dl:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = model(x)
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

        self.log_test(name, {
            "overall_acc": overall_acc,
            "tpr": tpr,
            "precision": precision,
            "f1": f1,
            "balanced_acc": balanced,
        })
        if print_vals:
            print(f"Overall per-point accuracy: {overall_acc:.3f}")
            print(f"Anomaly recall (TPR):       {tpr:.3f}")
            print(f"Precision:                  {precision:.3f}")
            print(f"F1 score:                   {f1:.3f}")
            print(f"Balanced accuracy:          {balanced:.3f}")

        return overall_acc,tpr,precision,f1,balanced





if __name__ == '__main__':
    PCS_test_accs = []
    reference_test_accs = []

    os.makedirs(os.path.dirname(filenamePCS), exist_ok=True)
    with open(filenamePCS, "a", encoding='utf-8') as f:
        f.write(f"TEST ON DATE: {time.strftime('%Y-%m-%d-%H-%M-%S')} \n")
    for i in range(num_repeats):
        run_seed = 1234 + i
        set_seed(run_seed)
        print('='*30 + ' Start Run {0}/{1} '.format(i + 1, num_repeats) + '='*30)
        t = SetTrainer(key)
        t.train_PCS()
        acc = t.test_PCS()
        PCS_test_accs.append(acc)
        print('=' * 30 + ' Finish Run {0}/{1} '.format(i + 1, num_repeats) + '=' * 30)
        with open(filenamePCS, "a", encoding='utf-8') as f:
            f.write(f"PCS Run: {i} Overall Accuracy: {acc[0]}, TPR: {acc[1]}, precision: {acc[2]}, f1: {acc[3]}, Balanced accuracy:  {acc[4]}\n")
        t.train_DeepSets()
        acc = t.test_DeepSets()
        reference_test_accs.append(acc)
        with open(filenamePCS, "a", encoding='utf-8') as f:
            f.write(f"DEEPSETS Run: {i} Overall Accuracy: {acc[0]}, TPR: {acc[1]}, precision: {acc[2]}, f1: {acc[3]}, Balanced accuracy:  {acc[4]}\n")
            f.write("\n\n")
    try:
        pcs_arr = np.array(PCS_test_accs)
        pcs_mean = pcs_arr.mean(axis=0)
        pcs_std = pcs_arr.std(axis=0)

        ref_arr = np.array(reference_test_accs)
        ref_mean = ref_arr.mean(axis=0)
        ref_std = ref_arr.std(axis=0)

        with open(filenamePCS, "a", encoding='utf-8') as f:
            f.write("\nPCS Results:")
            f.write(f"Overall Accuracy: {pcs_mean[0]:.4f} ± {pcs_std[0]:.4f}\n")
            f.write(f"TPR:  {pcs_mean[1]:.4f} ± {pcs_std[1]:.4f}\n")
            f.write(f"PRECISION:     {pcs_mean[2]:.4f} ± {pcs_std[2]:.4f}\n")
            f.write(f"F1:        {pcs_mean[3]:.4f} ± {pcs_std[3]:.4f}\n")
            f.write(f"Balanced accuracy:       {pcs_mean[4]:.4f} ± {pcs_std[4]:.4f}\n")

            f.write("\nDEEPSETS Results:")
            f.write(f"Overall Accuracy: {ref_mean[0]:.4f} ± {ref_std[0]:.4f}\n")
            f.write(f"TPR:  {ref_mean[1]:.4f} ± {ref_std[1]:.4f}\n")
            f.write(f"PRECISION:     {ref_mean[2]:.4f} ± {ref_std[2]:.4f}\n")
            f.write(f"F1:        {ref_mean[3]:.4f} ± {ref_std[3]:.4f}\n")
            f.write(f"Balanced accuracy:       {ref_mean[4]:.4f} ± {ref_std[4]:.4f}\n")

    except:
        print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(PCS_test_accs), np.std(PCS_test_accs)))




