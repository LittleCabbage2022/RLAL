# 文件路径: src/agents/env_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvModel(nn.Module):
    """
    Epoch embedding + GRU environment model.
    - embedding_table: nn.Embedding(num_epochs, embed_dim)
    - GRU: takes sequence of embeddings [0..et] and returns hidden state z (dim embed_dim)
    API:
      - forward(seq_idx_tensor) -> z (batch_size x embed_dim)
      - get_z_for_epoch(et) -> z tensor (embed_dim,)  -- computes using embeddings[0:et+1]
      - get_z_batch(epoch_indices) -> (batch_size x embed_dim)
    Notes:
      - For efficiency, you may cache z for computed epochs externally.
    """
    def __init__(self, num_epochs=200, embed_dim=128, gru_hidden=128, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_epochs, embed_dim)
        # GRU will process sequence of embeddings, return hidden state
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True)
        # project GRU hidden to z_dim (we keep z_dim == gru_hidden)
        self.z_dim = gru_hidden
        self.to(self.device)

    def forward(self, emb_seq_idxs):
        """
        emb_seq_idxs: LongTensor of shape (batch, seq_len) containing epoch indices (0..num_epochs-1)
        returns: z tensor of shape (batch, z_dim)
        """
        emb_seq_idxs = emb_seq_idxs.to(self.device)
        emb = self.embedding(emb_seq_idxs)  # (batch, seq_len, embed_dim)
        out, h = self.gru(emb)  # h: (1, batch, hidden)
        z = h.squeeze(0)  # (batch, hidden)
        return z

    def get_z_for_epoch(self, et):
        """
        Compute z for single epoch index et (int)
        uses sequence [0,1,...,et] as input
        """
        if et < 0:
            et = 0
        seq = torch.arange(0, et + 1, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
        with torch.no_grad():
            z = self.forward(seq)  # (1, z_dim)
        return z.squeeze(0)  # (z_dim,)

    def get_z_batch(self, epoch_idx_array):
        """
        epoch_idx_array: iterable of epoch indices (batch_size,)
        returns: (batch_size, z_dim)
        """
        # For simplicity we compute each sequence individually (not super efficient).
        # Option: optimize by grouping identical lengths.
        zs = []
        for et in epoch_idx_array:
            zs.append(self.get_z_for_epoch(int(et)))
        return torch.stack(zs, dim=0).to(self.device)
