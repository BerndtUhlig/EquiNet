import torch, platform, csv, os, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

import numpy as np
import models
import S3DISDataset
from torch.utils.data import Dataset, DataLoader, Subset
from CSVHelper import init_metric_csv, append_metric_csv

d = 3 # Individual Set Item Channels
N = 512 # Set Size
DS = 4096 # Dataset Size
batch_size = 8 # Batch Size
learning_rate = 1e-4
num_epochs = 1000
num_repeats = 3
filenamePCS = f"./ReferenceImpls/S3DIS/results/test_accs_{time.strftime('%Y-%m-%d-%H-%M-%S')}.txt"
num_classes = 13
metrics_PCS_csv = f"./ReferenceImpls/S3DIS/results/val_metrics_PCS_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"
metrics_DEEPSETS_csv = f"./ReferenceImpls/S3DIS/results/val_metrics_DEEPSETS_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv"

class S3DISTrainer:
    def __init__(self):
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch built with CUDA:", torch.version.cuda)
        print("OS:", platform.platform())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PCSNetwork = models.PointNetModel(N, 6,64,num_classes).to(self.device)
        self.referenceNetwork = models.DefaultModel(N,6,64,num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.PCSNetwork.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(100,num_epochs,100)),gamma=0.1)
        self.trainDataset = S3DISDataset.S3DISProcessedDataset("./ReferenceImpls/S3DIS/processed", "train", npoints=N)
        self.testDataset = S3DISDataset.S3DISProcessedDataset("./ReferenceImpls/S3DIS/processed", "test",npoints=N)
        self.valDataset = S3DISDataset.S3DISProcessedDataset("./ReferenceImpls/S3DIS/processed", "val", npoints=N)
        # fixed training-eval subset
        self.trainEvalDataset = self.make_fixed_subset(
            self.trainDataset,
            fraction=0.1,   # 10% of training set
            seed=42
        )
    def make_fixed_subset(self, dataset, fraction=0.1, seed=42):
        n = len(dataset)
        k = max(1, int(n * fraction))

        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=k, replace=False)
        indices = np.sort(indices)

        return Subset(dataset, indices.tolist())

    def train_PCS(self,run,):
        init_metric_csv(metrics_PCS_csv)
        self.PCSNetwork.train()
        optimizer = optim.AdamW(
            self.PCSNetwork.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(range(400, num_epochs, 400)), gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()

        dl = DataLoader(
            self.trainDataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

        best_val_f1 = -1.0
        best_state = None
        best_epoch = -1
        for epoch in trange(num_epochs, desc="PCS Epochs"):
            self.PCSNetwork.train()
            for x, y, _, _ in tqdm(dl, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.PCSNetwork(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.PCSNetwork.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            if epoch % 20 == 0:
                val_metrics = self.evaluate_model(self.PCSNetwork, self.valDataset, num_classes=num_classes)
                val_f1 = val_metrics[3]
                train_metrics = self.evaluate_model(self.PCSNetwork, self.trainEvalDataset, num_classes=num_classes)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in self.PCSNetwork.state_dict().items()}
                    best_epoch = epoch + 1
                append_metric_csv(metrics_PCS_csv,run,"PCS","val",epoch+1,val_metrics[0],val_metrics[1],val_metrics[2],val_metrics[3])
                append_metric_csv(metrics_PCS_csv, run, "PCS", "train", epoch + 1, train_metrics[0], train_metrics[1],
                                  train_metrics[2], train_metrics[3])
        if best_state is not None:
            with open(filenamePCS, "a", encoding='utf-8') as f:
                f.write(f"BEST VALIDATION AT EPOCH: {best_epoch}\n")
            self.PCSNetwork.load_state_dict(best_state)



    def test_PCS(self,run):
        test_metrics = self.evaluate_model(self.PCSNetwork, self.testDataset, num_classes=13, print_vals=True)
        append_metric_csv(metrics_PCS_csv, run, "PCS", "test", 0, test_metrics[0], test_metrics[1],
                          test_metrics[2], test_metrics[3])
        return test_metrics

    def train_regular(self,run):
        init_metric_csv(metrics_DEEPSETS_csv)
        self.referenceNetwork.train()
        optimizer = optim.AdamW(
            self.referenceNetwork.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(range(400, num_epochs, 400)), gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()

        dl = DataLoader(
            self.trainDataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

        best_val_f1 = -1.0
        best_state = None
        best_epoch = -1

        for epoch in trange(num_epochs, desc="Reference Epochs"):
            self.referenceNetwork.train()
            for x, y, _, _ in tqdm(dl, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.referenceNetwork(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.referenceNetwork.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            if epoch % 20 == 0:
                val_metrics = self.evaluate_model(self.referenceNetwork, self.valDataset, num_classes=num_classes)
                val_f1 = val_metrics[3]
                train_metrics = self.evaluate_model(self.referenceNetwork,self.trainEvalDataset, num_classes=num_classes)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in self.referenceNetwork.state_dict().items()}
                    best_epoch = epoch + 1
                append_metric_csv(metrics_DEEPSETS_csv, run, "DeepSets", "val", epoch + 1, val_metrics[0], val_metrics[1],
                                  val_metrics[2], val_metrics[3])
                append_metric_csv(metrics_DEEPSETS_csv, run, "DeepSets", "train", epoch + 1, train_metrics[0],
                                  train_metrics[1],
                                  train_metrics[2], train_metrics[3])

        if best_state is not None:
            with open(filenamePCS, "a", encoding='utf-8') as f:
                f.write(f"BEST VALIDATION AT EPOCH: {best_epoch}\n")
            self.referenceNetwork.load_state_dict(best_state)

    def test_regular(self,run):
        test_metrics = self.evaluate_model(self.referenceNetwork, self.testDataset, num_classes=13, print_vals=True)
        append_metric_csv(metrics_DEEPSETS_csv, run, "DeepSets", "test", 0, test_metrics[0], test_metrics[1],
                          test_metrics[2], test_metrics[3])
        return test_metrics

    def evaluate_model(self, model, dataset, num_classes=13, print_vals=False):
        model.eval()
        conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

        with torch.no_grad():
            for x, y, _, _ in dl:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = model(x)              # [B, C, N]
                preds = logits.argmax(dim=1)   # [B, N]

                for t, p in zip(y.view(-1), preds.view(-1)):
                    conf[t.long(), p.long()] += 1

        conf = conf.cpu()
        total = conf.sum().item()
        correct = conf.diag().sum().item()
        overall_acc = correct / max(total, 1)

        precisions, recalls, f1s = [], [], []
        for c in range(num_classes):
            tp = conf[c, c].item()
            fp = conf[:, c].sum().item() - tp
            fn = conf[c, :].sum().item() - tp

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            if print_vals:
                print(
                    f"Class {c}: "
                    f"precision={precision:.3f}, "
                    f"recall={recall:.3f}, "
                    f"f1={f1:.3f}"
                )




        macro_precision = sum(precisions) / num_classes
        macro_recall = sum(recalls) / num_classes
        macro_f1 = sum(f1s) / num_classes
        if print_vals:
            print()
            print(f"Macro precision: {macro_precision:.3f}")
            print(f"Macro recall:    {macro_recall:.3f}")
            print(f"Macro F1:        {macro_f1:.3f}")
        return overall_acc, macro_precision, macro_recall, macro_f1




if __name__ == '__main__':
    PCS_test_accs = []
    reference_test_accs = []

    os.makedirs(os.path.dirname(filenamePCS), exist_ok=True)
    with open(filenamePCS, "a", encoding='utf-8') as f:
        f.write(f"TEST ON DATE: {time.strftime('%Y-%m-%d-%H-%M-%S')} \n")
    for i in range(num_repeats):
        print('='*30 + ' Start Run {0}/{1} '.format(i + 1, num_repeats) + '='*30)
        t = S3DISTrainer()
        t.train_PCS(i+1)
        acc = t.test_PCS(i+1)
        PCS_test_accs.append(acc)
        print('=' * 30 + ' Finish Run {0}/{1} '.format(i + 1, num_repeats) + '=' * 30)
        with open(filenamePCS, "a", encoding='utf-8') as f:
            f.write(f"PCS Run: {i+1} Overall Accuracy: {acc[0]}, macro_precision: {acc[1]}, macro_recall: {acc[2]}, macro_f1: {acc[3]}\n")
        t.train_regular(i+1)
        acc = t.test_regular(i+1)
        reference_test_accs.append(acc)
        with open(filenamePCS, "a", encoding='utf-8') as f:
            f.write(f"REFERENCE Run: {i+1} Overall Accuracy: {acc[0]}, macro_precision: {acc[1]}, macro_recall: {acc[2]}, macro_f1: {acc[3]}\n")
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
            f.write(f"Macro Precision:  {pcs_mean[1]:.4f} ± {pcs_std[1]:.4f}\n")
            f.write(f"Macro Recall:     {pcs_mean[2]:.4f} ± {pcs_std[2]:.4f}\n")
            f.write(f"Macro F1:         {pcs_mean[3]:.4f} ± {pcs_std[3]:.4f}\n")

            f.write("\nREFERENCE Results:")
            f.write(f"Overall Accuracy: {ref_mean[0]:.4f} ± {ref_std[0]:.4f}\n")
            f.write(f"Macro Precision:  {ref_mean[1]:.4f} ± {ref_std[1]:.4f}\n")
            f.write(f"Macro Recall:     {ref_mean[2]:.4f} ± {ref_std[2]:.4f}\n")
            f.write(f"Macro F1:         {ref_mean[3]:.4f} ± {ref_std[3]:.4f}\n")

    except:
        print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(PCS_test_accs), np.std(PCS_test_accs)))