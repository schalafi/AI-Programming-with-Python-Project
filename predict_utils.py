import torch as th 
from train_utils import build_model
from checkpoint_utils import load_checkpoint



class Predictor:

    def __init__(self, checkpoint_path:str):
        """
        Args:
            checkpoint_path: path to the checkpoint file
        """
        self.checkpoint_path = checkpoint_path

        # 1 load the checkpoint
        print('Loading checkpoint: {}'.format(checkpoint_path))
              
        epoch, loss, model_name, n_hidden_units, n_classes,model,optimizer  = load_checkpoint(
                    path  = self.checkpoint_path
                    device_name = 'gpu')
        self.model_name = model 
        self.model = model 
                              

        # 2 preprocess the input image
        # 3 predict on the input image 

        

    