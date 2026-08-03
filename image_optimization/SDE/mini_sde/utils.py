from torchvision import utils
import torch


def save_image_grid(tensor, filename, nrow=8):
    # tensor expected in [0,1]
    utils.save_image(tensor, filename, nrow=nrow, padding=2)

def get_data_scaler(x):
    return x * 2. - 1.

def get_data_inverse_scaler(x):
    return (x + 1.) / 2.

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def save_checkpoint(ckpt_dir, model, ema):
    saved_state = {
      'model': model.state_dict(),
      'ema': ema.state_dict(),
    }
    torch.save(saved_state, ckpt_dir)