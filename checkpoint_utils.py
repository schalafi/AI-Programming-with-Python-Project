import os

import torch as th
from model_utils import build_model

def save_checkpoint(model: th.nn.Module,
                    optimizer:th.optim.Optimizer,
                    epoch: int,
                    loss: float ,
                    path:str,
                    model_name: str,
                    n_hidden_units: int,
                    n_classes: int
                    ):
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
        model_name (str): the name of the network architecture used to build model
        n_hidden_units (int): the number of hidden units in the network's classifier.
        n_classes (int): the number of classes in the network will output.
    Returns:
        None
    """
    # if the dirs in the path does not exist it create them
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        # save only the classifier portion of the model
        'model_state_dict': model.classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_name': model_name,
        'n_hidden_units': n_hidden_units,
        'n_classes': n_classes,
        'class_to_idx': model.classifier.class_to_idx
    }
    
    print("\033[94m {}\033[00m" .format('Creating checkpoint: '+ path ))
    print()

    th.save(checkpoint, path)


def load_checkpoint(path:str ,
                    device_name:str):
    """
    Build the model given by model_name (from the checkpoint)
    Load the parameters into the model 
    Load the state of the optimizer 

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint's optimizer state.
        path (str): The file path of the saved checkpoint.
        device (str, optional): The target device to load the model (e.g., 'cuda' or 'cpu').

    Returns:
        epoch (int): The epoch at which training was left off.
        loss (float): The training loss at the checkpoint.
        model_name (str): The model architecture the model has
        n_hidden_units (int): The number of hidden units in the model's classifier.
        n_classes (int): The number of classes in the model's classifier.
        model (torch.nn.Module): The PyTorch model loaded from the checkpoint.
        optimizer (torhc.optim.Optimizer)
    """

    device = None
    if device_name is not None:
        if not device_name in ['cuda', 'cpu']:
            raise ValueError('Invalid device name: {}'.format(device_name))
        selected_device = 'cpu'
        if device_name == 'cuda':
            if th.cuda.is_available():
                selected_device = 'cuda'
        
        device = th.device(selected_device)
    else:    
        device = th.device('cpu')
        
    print("\033[94m {}\033[00m" .format('Loading checkpoint: '+ path ))
    print("Device: ", device)
    checkpoint = th.load(path, map_location=device)
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model_name = checkpoint['model_name']
    n_hidden_units = checkpoint['n_hidden_units']
    n_classes = checkpoint['n_classes']
    class_to_idx = checkpoint['class_to_idx']

    # build the model from the model_name (only the architecture)
    # we also need model params
    model = build_model(model_name = model_name,
                        device_name = device_name,
                        train_dataset=None,
                        n_hidden_units=n_hidden_units,
                        n_classes=n_classes)

    def build_optimizer(model):
        optimizer = th.optim.Adam(
            #Only pass classifier params
            model.classifier.parameters(),
            lr = 0.001,
            weight_decay = 0.001)
        return optimizer

    optimizer = build_optimizer(model)

    #load the params into the model
    model.classifier.load_state_dict(checkpoint['model_state_dict'])
    # Ensure the model is on the correct device
    model.to(device)
    model.classifier.class_to_idx = class_to_idx

    # load the state of the optimizer 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return epoch, loss, model_name, n_hidden_units, n_classes,model,optimizer