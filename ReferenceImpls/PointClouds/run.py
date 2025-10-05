import torch, platform, csv, os, time
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler
from torch.autograd import Variable
from tqdm import tqdm, trange
import numpy as np

import classifier
import modelnet
import logging

log_every = 10     # <— write to CSV every N training steps
log_path = 'training_log_PCS_1000_03.csv'
#################### Settings ##############################
# PCS 1000 Test Accuracy 0.852
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

def current_lr(optimizer):
    return optimizer.param_groups[0].get('lr', None)

#################### Settings ##############################
num_epochs = 1000
batch_size = 64
downsample = 100  #For 5000 points use 2, for 1000 use 10, for 100 use 100
network_dim = 256 #For 5000 points use 512, for 1000 use 256, for 100 use 256
num_repeats = 1    #Number of times to repeat the experiment
data_path = 'ModelNet40_cloud.h5'
#################### Settings ##############################


class PointCloudTrainer(object):
    def __init__(self):
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch built with CUDA:", torch.version.cuda)
        print("OS:", platform.platform())
        #Data loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_fetcher = modelnet.ModelFetcher(data_path, batch_size, downsample, do_standardize=True, do_augmentation=True)
        sample_size = int(10000/downsample)
        #Setup network
        self.D = classifier.DPCSTanh(sample_size,network_dim, pool='max').to(self.device)
        self.L = nn.CrossEntropyLoss().to(self.device)
        #self.optimizer = optim.Adam([{'params':self.D.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)

        self.optimizer = optim.AdamW([{'params':self.D.parameters()}], lr=1e-4, weight_decay=1e-5, eps=1e-8) # optionally use this for 5000 points case, but adam with scheduler also works
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400, num_epochs, 400)),
                                       gamma=0.1)
        # (Optional) small perf wins
        torch.backends.cudnn.benchmark = True
        self.logger = CSVLogger(
            log_path,
            fieldnames=[
                "timestamp", "epoch",
                "loss", "running_train_acc", "lr"
            ]
        )
    def train(self):
        self.D.train()
        loss_val = float('inf')
        for j in trange(num_epochs, desc="Epochs: "):
            counts = 0
            sum_acc = 0.0
            train_data = self.model_fetcher.train_data(loss_val)
            for x, _, y in train_data:
                counts += len(y)
                X = torch.as_tensor(x, device= self.device, dtype=torch.float32)
                Y = torch.as_tensor(y,device=self.device, dtype=torch.long)
                val0 = X[0]

                self.optimizer.zero_grad()
                f_X = self.D(X)
                loss = self.L(f_X, Y)
                loss_val = float(loss.detach())
                max = f_X.argmax(dim=1)
                sum_acc += (max == Y).float().sum().item()
                train_data.set_description('Train loss: {0:.4f}'.format(loss_val))
                loss.backward()
                classifier.clip_grad(self.D, 5)
                self.optimizer.step()
                del X,Y,f_X,loss
            train_acc = sum_acc/counts
            self.scheduler.step()
            if j%10==9:
                self.logger.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": j + 1,
                    "loss": loss_val,
                    "running_train_acc": train_acc,
                    "lr": current_lr(self.optimizer),
                })
                tqdm.write('After epoch {0} Train Accuracy: {1:0.3f} '.format(j+1, train_acc))

    def test(self):
        self.D.eval()
        counts = 0
        sum_acc = 0.0
        for x, _, y in self.model_fetcher.test_data():
            counts += len(y)
            X = Variable(torch.cuda.FloatTensor(x))
            Y = Variable(torch.cuda.LongTensor(y))
            f_X = self.D(X)
            max = (f_X.max(dim=1)[1])
            sum_acc += (max == Y).float().sum().item()
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('Final Test Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc

if __name__ == "__main__":
    test_accs = []
    for i in range(num_repeats):
        print('='*30 + ' Start Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
        t = PointCloudTrainer()
        t.train()
        acc = t.test()
        test_accs.append(acc)
        print('='*30 + ' Finish Run {0}/{1} '.format(i+1, num_repeats) + '='*30)
    print('\n')
    if num_repeats > 2:
        try:
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            print(f"Test accuracy: {mean_acc:0.2f} ± {std_acc:0.3f}")
        except:
            print('Test accuracy: {0:0.2f} +/-  {0:0.3f} '.format(np.mean(test_accs), np.std(test_accs)))
