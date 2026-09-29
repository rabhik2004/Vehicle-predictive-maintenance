"""Sequence models (LSTM / GRU) for failure prediction. Requires PyTorch.

Usage:  python lstm_model.py --arch lstm      (or --arch gru)
Each sample = last SEQ_LEN raw sensor readings of one vehicle; label = failure
within the next 20 readings (same label as the tabular models).
NOTE: written to run standalone; PyTorch was not installable in the build
sandbox, so this file was not executed there - run it locally.
"""
import argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from generate_data import generate, SENSORS

SEQ_LEN = 30


def make_sequences(df):
    X, y, g = [], [], []
    for vid, d in df.groupby("vehicle_id"):
        v, l = d[SENSORS].values, d.label.values
        for i in range(SEQ_LEN, len(d)):
            X.append(v[i - SEQ_LEN:i]); y.append(l[i - 1]); g.append(vid)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(g)


class SeqNet(nn.Module):
    def __init__(self, n_in, arch="lstm", hidden=64):
        super().__init__()
        rnn = nn.LSTM if arch == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def main(arch, epochs=15):
    torch.manual_seed(42)
    df = generate(seed=42)
    X, y, g = make_sequences(df)
    tr, te = next(GroupShuffleSplit(1, test_size=.25, random_state=42).split(X, y, g))
    sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
    f = lambda a: torch.tensor(sc.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape), dtype=torch.float32)
    Xtr, Xte, ytr, yte = f(X[tr]), f(X[te]), torch.tensor(y[tr]), y[te]
    net = SeqNet(X.shape[-1], arch)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / (ytr == 1).sum()))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; opt.zero_grad()
            lossf(net(Xtr[b]), ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): p = torch.sigmoid(net(Xte)).numpy()
        print(f"epoch {ep+1:2d}  AUC={roc_auc_score(yte, p):.4f}  F1={f1_score(yte, p >= .5):.4f}")
    print(f"{arch.upper()} final: precision={precision_score(yte, p>=.5):.3f} "
          f"recall={recall_score(yte, p>=.5):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="lstm", choices=["lstm", "gru"])
    main(ap.parse_args().arch)
