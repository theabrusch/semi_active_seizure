from src.visualization import utils_vis
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch


def spectral_amplitude_perturbation(model, data, n_iterations, device):
    '''
    Function for applying the spectral amplitude perturbation

    model: (nn.Module)
        Pytorch model to use for predictions
    data: (np.NdArray)
        Numpy array with shape (samples, channels, time)
    n_iterations: (int)
        Number of perturbations to run

    Implementation is based on https://github.com/robintibor/braindecode/blob/62c9163b29903751a1dff08e243fcfa0bf7a7118/braindecode/visualization/perturbation.py#L147
    References
    ----------
    .. [EEGDeepLearning] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    '''

    orig_dataset = TensorDataset(torch.Tensor(data))
    orig_dataloader = DataLoader(orig_dataset, batch_size=1024, shuffle = False)
    
    # get original evaluation
    orig_pred = model_eval(model, orig_dataloader, device)

    # convert to frequency spectrum
    fft_input = np.fft.rfft(data, n=data.shape[2], axis=2)
    amps = np.abs(fft_input)
    phases = np.angle(fft_input)

    pert_corrs = 0

    for i in range(n_iterations):
        # Compute perturbed inputs
        amps_pert, phases_pert, pert_vals = amp_perturbation_additive(amps, phases)
        fft_pert = amps_pert * np.exp(1j * phases_pert)
        inputs_pert = np.fft.irfft(fft_pert, n=data.shape[2], axis=2).astype(
            np.float32
        )

        # convert to dataloader
        pert_dataset = TensorDataset(torch.Tensor(inputs_pert))
        pert_dataloader = DataLoader(pert_dataset, batch_size=1024, shuffle = False)

        # get perturbed outputs
        pert_output = model_eval(model, pert_dataloader, device)
        
        # get difference between perturbed and original outputs
        out_diff = pert_output - orig_pred

        pert_corrs_tmp = utils_vis.wrap_reshape_apply_fn(
                utils_vis.corr, pert_vals, out_diff, axis_a=(0,), axis_b=(0)
            )
        pert_corrs += pert_corrs_tmp

    perturbation = pert_corrs/n_iterations
    return perturbation



def model_eval(model, dataloader, device):
    pred = None
    model.eval()

    for batch in dataloader:
        x = batch[0].float().to(device)
        _, features = model(x, return_features = True)
        if pred is None:
            pred = features.detach().cpu().numpy()
        else:
            pred = np.append(pred, features.detach().cpu().numpy(), axis = 0)

    return pred


def amp_perturbation_additive(amps, phases, rng=None):
    """Takes amplitudes and phases of BxCxF with B input, C channels, F frequencies
    Adds additive noise N(0,0.02) to amplitudes
    Parameters
    ----------
    amps : numpy array
        Spectral amplitude
    phases : numpy array
        Spectral phases (not used)
    rng : object
        Random Seed
    Returns
    -------
    amps_pert : numpy array
        Scaled amplitudes
    phases_pert : numpy array
        Input phases (not modified)
    pert_vals : numpy array
        Amplitude noise
    """
    if rng is None:
        rng = np.random.RandomState()
    amp_noise = rng.normal(0, 1, amps.shape).astype(np.float32)
    amps_pert = amps + amp_noise
    amps_pert[amps_pert < 0] = 0
    amp_noise = amps_pert - amps
    return amps_pert, phases, amp_noise