from torchvision import utils


def save_image_grid(tensor, filename, nrow=8):
    # tensor expected in [0,1]
    utils.save_image(tensor, filename, nrow=nrow, padding=2)