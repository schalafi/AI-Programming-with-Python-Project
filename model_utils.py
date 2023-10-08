import torch as th
from torchvision import datasets
from torchvision import models



def build_model(model_name:str,
                device_name:str,
                n_hidden_units: int,
                train_dataset:datasets.ImageFolder = None,
                n_classes: int = None 
                )-> th.nn.Module:
    """
    Args:
        model_name: name of the model
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
    model = models_available.get(model_name,None)

    if not model:
        raise ValueError(f"Model {model_name} not supported\n Models supported{list(models_available.keys())}")
    
    if device_name not in ['cpu','cuda']:
        raise ValueError(f"Device {device_name} not supported")
    
    if device_name == 'cuda':
        if not  th.cuda.is_available():
            print("\033[93m" + "Warning: GPU not available" + "\033[0m")
            print('CPU will be used')
            device_name  = 'cpu'

    device = th.device(device_name)
    print("Device: ", device)

    #freeze model params
    # Turn off gradient computation on pretrained network
    for param in model.parameters():
        param.requires_grad = False 
    
    # get the number of classes  from the train_dataset or from the n_classes param
    #classes_names = train_dataset.classes

    num_classes = None
    if train_dataset:
        classes = train_dataset.classes
        num_classes = len(classes)
    if n_classes:
        num_classes = n_classes

    print("Number of classes: ", num_classes)

    if num_classes is None:
        raise ValueError("Number of classes is required or you must pass a train_dataset")

    #get number of input neurons depending the model
    n_inputs = None 

    n_inputs_dict = {
        'vgg16': 25088,
        'alexnet': 9216,
        'densenet121': 1024
    }
    
    if model_name == 'alexnet':
        n_inputs = model.classifier[1].in_features

    elif model_name == 'vgg16':
        n_inputs = model.classifier[0].in_features
    elif model_name == 'densenet121':
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
