import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from .dataset import WindowedDS, FEATURES
from .model import GRUGauss


class GaussianNLL(nn.Module):
    def forward(self, y_pred_mu, y_pred_var, y_true):
        return 0.5 * (torch.log(y_pred_var) + (y_true - y_pred_mu)**2 / y_pred_var).mean()


def train_on_df(df, lookback=30, epochs=8, lr=1e-3, batch=64, device="cpu"):
    ds = WindowedDS(df, lookback=lookback)
    n_val = max(64, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)


    model = GRUGauss(n_feat=len(FEATURES)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = GaussianNLL()


    hist = {"train_nll":[], "val_nll":[], "val_mae":[], "val_diracc":[]}


    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, var = model(xb)
            loss = criterion(mu, var, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            # eval
        model.eval(); tl, vl, mae, hits, n = 0,0,0,0,0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, var = model(xb)
                vl += criterion(mu, var, yb).item()*len(xb)
                mae += (mu - yb).abs().sum().item()
                hits += ((mu>=0) == (yb>=0)).sum().item()
                n += len(xb)
        hist["val_nll"].append(vl/n)
        hist["val_mae"].append(mae/n)
        hist["val_diracc"].append(hits/n)
        print(f"ep {ep+1}: val_nll={vl/n:.4f} val_mae={mae/n:.5f} dir_acc={hits/n:.3f}")


    return model, hist