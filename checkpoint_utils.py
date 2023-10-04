import os

import torch as th


def save_checkpoint(model: th.nn.Module,
                    optimizer:th.optim.Optimizer,
                    epoch: int,
                    loss: float ,
                    path:str):
    """
    Save a checkpoint of a PyTorch model for inference or resuming training.
    path must be  the path of the model checkpoint file
    example: ./models/my_model.pth
    
    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        epoch (int): Number of epochs model has been trained.
        loss (float): The current training loss.
        path (str): The file path to save the checkpoint.

    Returns:
        None
    """
    # if the dirs in the path does not exist it create them
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    print("\033[94m {}\033[00m" .format('Creating checkpoint: '+ path ))
    print()

    th.save(checkpoint, path)
