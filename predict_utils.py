from train_utils import build_model
from checkpoint_utils import load_checkpoint



class Predictor:

    def __init__(self, model_path):
        self.model = build_model()
        self.model.load_weights(model_path)

        ### TODO:
        #create a method to build the model given the model name
        # then  load_checkpoint 
        # Modify save_checkpoint to also save the model name

        # build the model

        # load the checkpoint 

        # preprocess the input image

        # predict on the input image 
