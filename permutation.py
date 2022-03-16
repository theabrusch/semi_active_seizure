from src.visualization import utils_vis, perturbation
from src.models import get_model
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

pert_vals = np.random.randn(100, 20, 500)
preds_diff = np.random.randn(100, 2)
pert_corrs_tmp = utils_vis.wrap_reshape_apply_fn(
            utils_vis.corr, pert_vals, preds_diff, axis_a=(0,), axis_b=(0)
        )
orig_dataset = TensorDataset(torch.Tensor(pert_vals))
orig_dataloader = DataLoader(orig_dataset, batch_size=1024)


model_dict = {'model': 'BaselineCNN', 'input_shape': (20,500), 'model_summary': False}
model = get_model.get_model(model_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pert_map = perturbation.spectral_amplitude_perturbation(model, pert_vals, 1, device)