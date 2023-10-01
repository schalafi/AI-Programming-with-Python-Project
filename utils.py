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



BATCH_SIZE = 64

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

def get_dataloaders():
    """
    Return a dictionary with 
    'train', 'valid', 'test' keys
    each with a dataloader

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
        'test': test_transforms'
    }


    # TODO: Load the datasets with ImageFolder
    #image_datasets = datasets.ImageFolder()
    train_dataset = datasets.ImageFolder(
        root = train_dir,
        transform = train_transforms
    )

    valid_dataset = datasets.ImageFolder(
        root = valid_dir,
        transform=valid_transforms
    )

    test_dataset = datasets.ImageFolder(
        root = test_dir,
        transform = test_transforms
    )

    print('Number of training examples: ', len(train_dataset))
    print('Number of validation examples: ', len(valid_dataset))
    print('Number of test examples: ', len(test_dataset))


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle= True, # Shuffle only for training
        )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    dataloaders = {'train': train_dataloader,
            'valid': valid_dataloader,
            'test': test_dataloader}

    # Get a batch of the data next(iter(train_dataloader))
    print_data_shape(dataloader = train_dataloader,name = 'train')

    print_data_shape(dataloader = valid_dataloader,name = 'valid')

    print_data_shape(dataloader = test_dataloader,name = 'test')    

    return dataloaders 


def get_data_sample(dataloader:th.utils.data.Dataloader ):

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

    with open('class_to_idx.json', 'r') as f:
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
        'densenet121': models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
    }

    model = None 
    model = models_available.get(name,None)

    if not model:
        raise ValueError(f"Model {name} not supported")
    
    if device_name not in ['cpu','gpu']:
        raise ValueError(f"Device {device} not supported")
    
    if device_name == 'gpu':
        if not  th.cuda.is_available():
            #print in yellow a warning indicating that the gpu is not available
            print("\033[93m" + "Warning: GPU not available" + "\033[0m")
            print('Cpu will be used')
            name  = 'cpu'

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

    # Replace the clasifier (last layer ) with a new one
    model.classifier = th.nn.Sequential(
    th.nn.BatchNorm1d(1024),
    #input_features must have the the same number of out_features as 
    #norm5 layer in the original pretrained net (DenseNet)
    th.nn.Linear(in_features=1024,
                    out_features=n_hidden_units,
                    bias = True),
    th.nn.ReLU(),
    th.nn.Dropout(p=0.07),
    th.nn.BatchNorm1d(256),
    th.nn.Linear(in_features =n_hidden_units,
                 out_features=num_classes,
                 bias = True),
    th.nn.LogSoftmax(dim = 1)).to(device)
    #print("Classifier (trainable layers): ", model.classifier)

    return model