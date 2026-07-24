import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import hdbscan
from tqdm import tqdm

class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
 
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
 
 
class _DECLayer(nn.Module):
    """
    Distribution soft q_{ij} via kernel t-Student
 
        q_{ij} = (1 + ||z_i - mu_j||^2)^{-1}
                 ─────────────────────────────
                 Σ_j' (1 + ||z_i - mu_j'||^2)^{-1}
    """
 
    def __init__(self, n_clusters: int, embedding_dim: int):
        super().__init__()
        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, embedding_dim)
        )
 
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, D)  cluster_centers: (K, D)
        diff    = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)  # (N,K,D)
        dist_sq = (diff ** 2).sum(dim=2)                               # (N,K)
        q       = 1.0 / (1.0 + dist_sq)
        q       = q / q.sum(dim=1, keepdim=True)
        return q
 
 
def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Distribution target p_{ij} = (q_{ij}^2 / f_j) / Σ_j' (q_{ij'}^2 / f_j')
    where f_j = Σ_i q_{ij}  
    """
    f = q.sum(dim=0, keepdim=True)
    p = (q ** 2) / f
    p = p / p.sum(dim=1, keepdim=True)
    return p
 
 
class DEC:
    """
    Deep Embedding Clustering (Xie et al., 2016).
 
    Pipeline:
      1. Autoencoder pretraining (MSE loss)
      2. Centroid initialization with KMeans on the embedding
      3. Joint fine-tuning of encoder + DECLayer (KL-divergence loss)
 
    GPU usage
    ---------
    If CUDA is available, it is used automatically.
    To force CPU: DEC(..., device='cpu')
 
    Parametri
    ----------
    n_clusters       : number of clusters
    embedding_dim    : latent space dimension (default 64)
    pretrain_epochs  : epochs for autoencoder pretraining
    dec_epochs       : maximum epochs for DEC fine-tuning
    batch_size       : batch size (increase if you have more VRAM)
    lr               : learning rate Adam
    tol              : early stopping threshold (fraction of changed labels)
    device           : 'cuda', 'cuda:0', 'cpu', or None (auto)
    num_workers      : workers for the DataLoader (0 on Windows)
    pin_memory       : True when using GPU (speeds up CPU→GPU transfer)
    """
 
    def __init__(
        self,
        n_clusters     : int,
        embedding_dim  : int   = 128,
        pretrain_epochs: int   = 100,
        dec_epochs     : int   = 150,
        batch_size     : int   = 4096,
        lr             : float = 2e-5,
        tol            : float = 5e-5,
        device         : str   = None,
        num_workers    : int   = 0,
        pin_memory     : bool  = None,
    ):
        self.n_clusters      = n_clusters
        self.embedding_dim   = embedding_dim
        self.pretrain_epochs = pretrain_epochs
        self.dec_epochs      = dec_epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.tol             = tol
        self.num_workers     = num_workers
 
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
 
        if pin_memory is None:
            self.pin_memory = self.device.type == "cuda"
        else:
            self.pin_memory = pin_memory
 
        print(f"  DEC device: {self.device}"
              + (" ✓ GPU" if self.device.type == "cuda" else " (CPU)"))
 
    def _make_loader(self, X_tensor: torch.Tensor, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(X_tensor),
            batch_size  = self.batch_size,
            shuffle     = shuffle,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
        )

    def _pretrain(self, loader: DataLoader, input_dim: int):
        self.autoencoder = _Autoencoder(input_dim, self.embedding_dim).to(self.device)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
 
        self.autoencoder.train()
        for epoch in tqdm(range(self.pretrain_epochs), desc="Pretraining"):
            epoch_loss = 0.0
            progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.pretrain_epochs}")
            for (xb,) in progress_bar:
                xb = xb.to(self.device, non_blocking=True)
                recon, _ = self.autoencoder(xb)
                loss = criterion(recon, xb)
                optimizer.zero_grad(set_to_none=True)   # slightly faster
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
 
            print(f"[Pretrain] {epoch+1:>3}/{self.pretrain_epochs}, loss={epoch_loss / len(loader):.6f}")
 
    @torch.no_grad()
    def _get_embeddings(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Extracts embeddings from the encoder in inference mode."""
        self.autoencoder.eval()
        loader = self._make_loader(X_tensor, shuffle=False)
        parts  = []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            _, z = self.autoencoder(xb)
            parts.append(z.cpu().numpy())
        return np.concatenate(parts)
 
    def _init_cluster_centers(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Initializes DEC centroids with KMeans on the embedding."""
        embeddings = self._get_embeddings(X_tensor)
 
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=27)
        init_labels = kmeans.fit_predict(embeddings)
 
        self.dec_layer = _DECLayer(self.n_clusters, self.embedding_dim).to(self.device)
        self.dec_layer.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(self.device)
 
        return init_labels
 
    def _reassign_empty_clusters(self, current_labels: np.ndarray, embeddings: np.ndarray):
        """
        If a cluster is empty, reinitialize its centroid
        at the point farthest from the most populated centroid.
        Returns True if at least one cluster was reinitialized.
        """
        reassigned = False
        centers = self.dec_layer.cluster_centers.data.cpu().numpy()

        for k in range(self.n_clusters):
            mask = current_labels == k
            if mask.sum() == 0:
                # Find the most populated cluster
                counts = np.bincount(current_labels, minlength=self.n_clusters)
                largest_k = counts.argmax()
                largest_center = centers[largest_k]

                # Take the point farthest from the dominant centroid
                pts = embeddings[current_labels == largest_k]
                dists = np.linalg.norm(pts - largest_center, axis=1)
                new_center = pts[dists.argmax()]

                self.dec_layer.cluster_centers.data[k] = torch.tensor(
                    new_center, dtype=torch.float32
                ).to(self.device)

                print(f"  [WARNING] Cluster {k} empty → reinitialized from cluster {largest_k}")
                reassigned = True

        return reassigned
    
    def _dec_finetune(
        self,
        X_tensor   : torch.Tensor,
        init_labels: np.ndarray,
    ) -> np.ndarray:
 
        # Fine-tuning only on encoder + DEC layer (decoder is no longer needed)
        optimizer = optim.Adam(
            list(self.autoencoder.encoder.parameters()) +
            list(self.dec_layer.parameters()),
            lr=self.lr * 0.1,
        )
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loader = self._make_loader(X_tensor, shuffle=False)
        prev_labels = init_labels.copy()
 
        for epoch in tqdm(range(self.dec_epochs), desc="Epochs"):
            self.autoencoder.train()
            self.dec_layer.train()
 
            all_q    = []
            total_loss = 0.0

            progress_bar = tqdm(
                loader,
                desc=f"[DEC] Epoch {epoch+1}/{self.dec_epochs}",
                leave=False
            )
            
            for (xb,) in progress_bar:
                xb = xb.to(self.device, non_blocking=True)
                _, z = self.autoencoder(xb)
                q    = self.dec_layer(z)
                p    = _target_distribution(q).detach()
 
                loss = kl_loss(q.log(), p)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
 
                total_loss += loss.item()
                all_q.append(q.detach().cpu())
 
            # Current labels from hard assignment
            current_labels = torch.cat(all_q).argmax(dim=1).numpy()

            embeddings = self._get_embeddings(X_tensor)
            self._reassign_empty_clusters(current_labels, embeddings)

            delta = (current_labels != prev_labels).mean()
 
            print(f"[DEC] {epoch+1:>3}/{self.dec_epochs}, loss={total_loss / len(loader):.6f}, label_change={delta:.6f}")
 
            if epoch > 10 and delta < self.tol:
                print(f"  Converged at epoch {epoch+1}  (Δ={delta:.2e} < tol={self.tol})")
                break
 
            prev_labels = current_labels
 
        return current_labels
 
    def fit_predict(self, X: np.ndarray):
        """
        Runs pretraining + initialization + fine-tuning and returns (centers, labels).
 
        Parameters
        ----------
        X : array (n_samples, n_features) — already normalized
 
        Returns
        -------
        cluster_centers : (n_clusters, n_features)  in the original space
        labels          : (n_samples,)              integers 0..n_clusters-1
        """
        scaler = StandardScaler()
        print("==> [Standard Scaler] Standard Scaler fit transform")
        X_scaled = scaler.fit_transform(X)

        X_tensor  = torch.tensor(X_scaled, dtype=torch.float32)
        input_dim = X.shape[1]
 
        print("==> [DEC] Pretraining autoencoder...")
        loader = self._make_loader(X_tensor, shuffle=True)
        self._pretrain(loader, input_dim)
 
        print("==> [DEC] KMeans initialization on embedding...")
        init_labels = self._init_cluster_centers(X_tensor)
 
        print("==> [DEC] Fine-tuning...")
        labels = self._dec_finetune(X_tensor, init_labels)
 
        centers = np.array([
            X[labels == k].mean(axis=0) for k in range(self.n_clusters)
        ])

        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n[CHECK] Active clusters: {len(unique)}/{self.n_clusters}")
        for u, c in zip(unique, counts):
            print(f"  Cluster {u}: {c} points ({100*c/len(labels):.1f}%)")

        if len(unique) < self.n_clusters:
            missing = set(range(self.n_clusters)) - set(unique)
            print(f"  [WARNING] Missing clusters: {missing}")
            
        return centers, labels