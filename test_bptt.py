import torch
from src.models.trm import SudokuTRMv2
from src.training import train_sudoku_trm_v2
from torch.utils.data import TensorDataset, DataLoader

# Build a tiny dummy dataset
torch.manual_seed(0)
x = torch.zeros(8, 9, 10); x[:,:,0] = 1   # (batch=8, cells=9, cell_dim=10)
y = torch.zeros(8, 9, dtype=torch.long)      # all-zero targets
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=4)

model = SudokuTRMv2(hidden_size=32, num_heads=2, num_layers=1, num_cells=9, num_digits=4, mlp_t=True)
print("Starting train...")
train_sudoku_trm_v2(
    model, loader, device=torch.device('cpu'),
    epochs=1, T=2, N_SUP=3, L_cycles=1, lr=1e-4, verbose=True
)
print('train_sudoku_trm_v2 FULL BPTT smoke test passed (N_SUP=3, T=2)')
