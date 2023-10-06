import time
import json 

#Data related
import torch as th 
from torch.utils.data import DataLoader
from torchvision import datasets
#using the newest transforms from version 2
from torchvision.transforms import v2

#Model related
from torchvision import models

import training_functions
import checkpoint_utils


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


def print_data_shape(dataloader,name = 'train'):
    X,y = next(iter(dataloader))
    print(name.capitalize() + ' info:')
    print("Input Features (X):")
    print("Batch size: ", X.shape[0])
    print("Shape: ", X.shape)
    print('Labels (y): ')
    print("Shape: ", y.shape)
    print()

def get_data(data_dir:str,batch_size:int):
    """
    Args:
        data_dir: directory containing train, validation and test data, 
        each one has is own subfolder.
        train, valid and test
        bathc_size: number of examples per batch

    Return a dictionary with 3 elements
    'train': (train_dataset,train_dataloader),
    'valid': (valid_dataset,valid_dataloader),
    'test': (test_dataset,test_dataloader) 

    """
    
    train_transforms = v2.Compose([
    #Data augmentation with transformations
    v2.RandomRotation(degrees =(0,180)),
    v2.RandomResizedCrop((224,224)),
    v2.RandomPerspective(distortion_scale=0.6,
                          p=0.2), #probability for applying it
    v2.RandomAffine(degrees=(30, 70),
                    translate=(0.1, 0.3),
                    scale=(0.5, 1.1)),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
     )
     ])

    # Do not use data augmentation transformations
    valid_transforms = v2.Compose([
        v2.Resize((224,224)),
        v2.ToTensor(),
        v2.Normalize(
            mean= [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    # Do not use data augmentation transformations
    test_transforms = v2.Compose([
        v2.Resize((224,224)),
        v2.ToTensor(),
        v2.Normalize(
            mean= [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )]
    )

    transforms = {
        'train': train_transforms,
        'valid': valid_transforms,
        'test': test_transforms
    }


    # TODO: Load the datasets with ImageFolder
    #image_datasets = datasets.ImageFolder()
    train_dataset = datasets.ImageFolder(
        root = data_dir + '/train',
        transform = train_transforms
    )

    valid_dataset = datasets.ImageFolder(
        root = data_dir + '/valid',
        transform=valid_transforms
    )

    test_dataset = datasets.ImageFolder(
        root = data_dir + '/test',
        transform = test_transforms
    )

    print('Number of training examples: ', len(train_dataset))
    print('Number of validation examples: ', len(valid_dataset))
    print('Number of test examples: ', len(test_dataset))


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle= True, # Shuffle only for training
        )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    dataloaders = {'train': (train_dataset,train_dataloader),
            'valid': (valid_dataset,valid_dataloader),
            'test': (test_dataset,test_dataloader)}

    # Get a batch of the data next(iter(train_dataloader))
    print_data_shape(dataloader = train_dataloader,name = 'train')

    print_data_shape(dataloader = valid_dataloader,name = 'valid')

    print_data_shape(dataloader = test_dataloader,name = 'test')    

    return dataloaders 


def get_data_sample(dataloader:th.utils.data.DataLoader ):

    data_iter = iter(dataloader)
    #Get a batch of data
    #images (batch_sizex3x224x224) Tensor of images
    #labels (batch_size) A vector 
    images,labels = next(data_iter)

    print("Sample shapes: ",images.shape, labels.shape )
    return images, labels 



def get_label_mapping():
    """
    Return a dictionary mapping from
        index to label name
    """

    with open('cat_to_name.json', 'r') as f:
        class_to_idx = json.load(f)
    return class_to_idx
    

def build_model(name:str,
                device_name:str,
                train_dataset:datasets.ImageFolder,
                n_hidden_units: int
                )-> th.nn.Module:
    """
    Args:
        name: name of the model
        device_name: the device to use (cpu or gpu)
        train_dataset: the training dataset
        n_hidden_units: number of hidden units in the classifier
    
    Returns:
        A torch model. The image classifier model
    """

    models_available = {
        'vgg16': models.vgg16(weights = models.VGG16_Weights.DEFAULT),
        'densenet121': models.densenet121(weights = models.DenseNet121_Weights.DEFAULT),
        #'resnet18': models.resnet18(weights = models.ResNet18_Weights.DEFAULT),
        'alexnet': models.alexnet(weights = models.AlexNet_Weights.DEFAULT),
    }

    model = None 
    model = models_available.get(name,None)

    if not model:
        raise ValueError(f"Model {name} not supported\n Models supported{list(models_available.keys())}")
    
    if device_name not in ['cpu','gpu']:
        raise ValueError(f"Device {device} not supported")
    
    if device_name == 'gpu':
        if not  th.cuda.is_available():
            print("\033[93m" + "Warning: GPU not available" + "\033[0m")
            print('Cpu will be used')
            device_name  = 'cpu'

    device = th.device(device_name)
    print("Device: ", device)

    #freeze model params
    # Turn off gradient computation on pretrained network
    for param in model.parameters():
        param.requires_grad = False 
    
    # get the number of classes 
    classes_names = train_dataset.classes
    num_classes =  len(classes_names)
    print("Number of classes: ", num_classes)

    #get number of input neurons depending the model
    n_inputs = None 

    n_inputs_dict = {
        'vgg16': 25088,
        'alexnet': 9216,
        'densenet121': 1024
    }
    
    if name == 'alexnet':
        n_inputs = model.classifier[1].in_features

    elif name == 'vgg16':
        n_inputs = model.classifier[0].in_features
    elif name == 'densenet121':
        n_inputs = model.classifier.in_features

    
    print("Number of outputs in penultimate layer: ", n_inputs)

    assert n_inputs in n_inputs_dict.values(), 'Error while passing number of inputs to classifier layer.'
    
    print("MODEL ARCH: ", model)

    # Replace the clasifier (last layer ) with a new one
    model.classifier = th.nn.Sequential(
    th.nn.BatchNorm1d(n_inputs),
    #input_features must have the the same number of out_features as 
    #norm5 layer in the original pretrained net (DenseNet)
    th.nn.Linear(in_features=n_inputs,
                    out_features=n_hidden_units,
                    bias = True),
    th.nn.ReLU(),
    th.nn.Dropout(p=0.07),
    th.nn.BatchNorm1d(n_hidden_units),
    th.nn.Linear(in_features =n_hidden_units,
                 out_features=num_classes,
                 bias = True),
    th.nn.LogSoftmax(dim = 1)).to(device)
    #print("Classifier (trainable layers): ", model.classifier)

    return model



class Trainer():
    """
    Train an image classification model 

    """
    
    def __init__(self,
                data_dir:str,
                save_dir:str,
                arch:str,
                learning_rate: float,
                hidden_units:int,
                epochs: int,
                gpu: bool,
                batch_size: int = 64,
                n_minibatches: int = float('inf')
                ):
        """
        Args:
            data_dir: the path to the data directory
            save_dir: the path to the save directory
            arch:     the name of the model architecture
            learning_rate: the learning rate
            hidden_units: number of hidden units in the classifier module
            epochs: number of epochs to train the network
            gpu: True if gpu is to be used, False otherwise
        
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.gpu = gpu
        self.batch_size = batch_size
        self.n_minibatches = n_minibatches
         
        device_name = 'gpu' if self.gpu else 'cpu'

        if device_name == 'gpu':
            if not  th.cuda.is_available():
                print("\033[93m" + "Warning: GPU not available" + "\033[0m")
                print('Cpu will be used')
                device_name  = 'cpu'
        
        self.device = th.device(device_name)
        print("Device: ", self.device)

        self.cat_to_name = get_label_mapping()
        self.model = None
        
        ### 1 Load the datasets
        print("\033[92m" + "Getting dataloaders" + "\033[0m")

        data = get_data(
            data_dir= self.data_dir,
            batch_size = self.batch_size)
        self.train_dataset, self.train_dataloader =  data['train']
        self.valid_dataset, self.valid_dataloader = data['valid']
        self.test_dataset,  self.test_dataloader = data['test']
        
        print()

        ### 2 Create the model 
        ### Use the given architecture and replace the classifier with a new trainable module (layers and activations)

        self.model = build_model(name = self.arch,
                device_name = 'gpu' if self.gpu else 'cpu',
                train_dataset = self.train_dataset,
                n_hidden_units =  self.hidden_units)
        print()

        ### 3 Define the loss function and optimizer
        # Use negative log likelihood loss
        self.criterion = th.nn.NLLLoss()

        self.optimizer = th.optim.Adam(
        #Only pass classifier params
        self.model.classifier.parameters(),
        lr = self.learning_rate,
        weight_decay = 0.001
        )
        print()

        ### 4 Train the model
        last_loss_value = self.train()
        print()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        ### 5 save checkpoint
        checkpoint_utils.save_checkpoint(
            model =  self.model,
            optimizer = self.optimizer,
            epoch = self.epochs,
            loss = last_loss_value,
            path = self.save_dir + '/' + timestamp +  '-checkpoint.pth' )
        
        print()
            
    def train(self):
        """
        Return last reported loss on training set
        """

        #print in purple 'Training ...  with n_minibatches = '
        print("\033[95m" + "Training ...  with n_minibatches=   " + "\033[0m" + str(self.n_minibatches))

        train_losses,train_accuracies, valid_losses,valid_accuracies=  training_functions.train(
            model = self.model,
            train_dataloader = self.train_dataloader,
            valid_dataloader = self.valid_dataloader,
            criterion = self.criterion,
            optimizer = self.optimizer,
            device = self.device,
            epochs = self.epochs,
            n_minibatches = self.n_minibatches,
           
        )


        return train_losses[-1]

        
