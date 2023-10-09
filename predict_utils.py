from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch as th 
from checkpoint_utils import load_checkpoint
import json



def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
    '''
    # Open the image using PIL
    img = Image.open(image)

    # Resize the image to 256x256 
    img.thumbnail((256, 256))

    # Crop the center 224x224 portion of the image
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = (img.width + 224) / 2
    bottom = (img.height + 224) / 2
    img = img.crop((left, top, right, bottom))
    print("SIZES: ", img.height,img.width)

    # Convert color values from range [0,255] to  the range [0, 1]
    np_image = np.array(img) / 255.0

    # Normalize the image using mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions to have color channels as the first dimension
    # from rowsxcolumnsxchannels to  channelsxrowsxcolumns
    # or can be view as HxWxC to CxHxW 
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image:np.ndarray, ax=None, title=None):
    """Imshow for numpy image."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

class Predictor:

    def __init__(self, checkpoint_path:str, 
                category_names : str = 'cat_to_name.json',
                gpu:bool = None  ):
        """
        Args:
            checkpoint_path: path to the checkpoint file
            category_names: path to the .json file with category mapping (int -> str)
            gpu: use GPU for inference or not
        
        Returns:
        
        """
        self.checkpoint_path = checkpoint_path
        self.category_names = category_names
        self.gpu = gpu   
        
        self.class_to_idx = None 

        # 1 load the checkpoint
        print('Loading checkpoint: {}'.format(checkpoint_path))
              
        epoch, loss, model_name, n_hidden_units, n_classes,model,optimizer = load_checkpoint(
                    path  = self.checkpoint_path,
                    device_name = 'cuda' if gpu else 'cpu')
        self.model_name = model_name 
        self.model = model 

        self.class_to_idx = self.model.classifier.class_to_idx
        
        selected_device = 'cpu'
        if gpu:
            if th.cuda.is_available():
                selected_device = 'cuda'
        else:
            selected_device = 'cpu'

        self.device = th.device(selected_device)
        

    # Use inference mode inside the function
    @th.inference_mode()
    def predict(self,image_path:str, topk:int=1):
        ''' 
        Predict the class (or classes) of an image using a trained deep learning model.
        Args:
            image_path
                Path of the input image

            checkpoint_path
                the checkpoint file (.pth) to retrieve the model 
                for prediction
            topk:
                return the top k classes 
        
        '''

        # 1 preprocess the input image
        # Get the image with correct shape and normalization
        preprocessed_image = process_image(image_path).astype(np.float32)
        # add batch dimension to get a tensor with shape [Batch_SizexCxHxW]
        preprocessed_image = preprocessed_image[None,:,:,:]
        # transform to th.Tensor
        input_X = th.from_numpy(preprocessed_image)
        input_X = input_X.to(self.device)

        # model to evaluation mode
        self.model.eval()

        def infer(X):
            """
            X: input tensor
                Run inference on the X input

            """
            outputs = self.model(X)
            
            # Compute the probabilities over classes
            # the model produces log probabilities
            # log probabilities -> probabilities
            probabilities = th.exp(outputs)

            # get the topk biggest values from probs
            #probabilities, location on the array (the classes)
            probs, indices = probabilities.topk(k=topk, dim=1) 

            # get one dimensional arrays
            probs = probs.squeeze()
            indices = indices.squeeze()

            return probs, indices

        # 2 predict on the input image
        probs,indices = infer(input_X)  
        # pass to cpu
        probs, indices  = probs.cpu().numpy(),indices.cpu().numpy()
        # get the mapping class -> index
        # mapping index -> class 
        idx_to_classes = {x:y for y,x in self.class_to_idx.items()}
        top_classes = [idx_to_classes[i] for i in indices] 

        with open(self.category_names, 'r') as f:
            cat_to_name = json.load(f)

        # 3 get class(es) name(s) 
        classes = []
    
        for i in range(topk):
            class_ = cat_to_name[top_classes[i]]
            classes.append(class_)

        return probs,classes

