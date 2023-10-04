import time 

import torch as th 


def train_step(model: th.nn.Module,
          dataloader: th.utils.data.DataLoader,
          criterion: th.nn.Module,
          optimizer: th.optim.Optimizer,
          device: th.device) -> None:
    """
    Receive a model, train it with the given optimizer
    by ONE epoch,
    using the `criterion` function to measure the performance of the model.
    The model will be trained on the `train_dataloader` and
    evaluated on the `valid_dataloader` during each epoch.
    The function will return the trained model.
    
    Args:
        model: The model to train.
        train_dataloader: The training data.
        criterion: The loss function.
        optimizer: The optimizer to use.
        device: The device to use.
    Model weigths are changed in place
    
    Returns:
    loss_value, accuracy 
        
    """
    
    # set model to train mode 
    model.train()

    # init 
    loss_value,accuracy = 0,0
    
    #start time 
    t0 = time.time()
    n_batches= len(dataloader)
    #Iterate over the pairs (X_i,y_i) (features, labels)
    for i, (X,y) in enumerate(dataloader):
        #if i >= 10:
        #    break
        print("Training on batch {} from {} batches".format(i,n_batches))

        #move the data to device
        #must be on the same device that model
        features,labels = X.to(device),y.to(device)
        
        # apply forward propagation
        # outputs is the log probabilities
        outputs = model(features)

        # compute loss
        loss= criterion(outputs, labels)

        # loss_value is a number
        batch_loss = loss.item()
        loss_value+=  batch_loss

        # reset gradient accumulation
        # gradient tensors to 0
        optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # update params. 
        # apply gradient descent
        optimizer.step()

        # Get the classes predicted by the model
        # logits -> classes 
        predicted_classes = th.argmax(
            th.softmax(outputs,dim = 1),
            dim= 1 
        )
        
        # Compute accuracy
        # log probabilities -> probabilities
        probs = th.exp(outputs)

        # get the topk biggest values from probs
        # values, indices = probs.topk(k=5, dim=1)  
        # we get top probability, top class (index)
        top_p, top_class = probs.topk(k=1, dim=1)

        # compare the predicted class (top_class) and the actual classes (labels)
        # transform (get a view) the labels into the shape of top class
        equals = top_class == labels.view(*top_class.shape)
        #train_accuracy += th.mean(equals.type(th.FloatTensor)).item()
        accuracy += th.mean(equals.type(th.float)).item()
        
    loss_value = loss_value / len(dataloader)
    
    accuracy = accuracy / len(dataloader)
    print('\033[102m' + 'New epoch completed!' + '\033[0m')
    print(f'Training Loss: {loss_value:.3f}')
    print(f'Training Accuracy: {100.0*accuracy:.3f}% ')
    print(f'Time Elapsed: {time.time() - t0:.3f} seconds')
    print()    

    return loss_value, accuracy


def validation_step(model: th.nn.Module,
          dataloader: th.utils.data.DataLoader,
          criterion: th.nn.Module,
          device: th.device) -> None:
    """
    Receive a model, run it  to validate on the given data.
    Use one epoch over the dataset on dataloader.
    The function will return accuracy and loss on the dataset
    
    Args:
        model: The model to train.
        train_dataloader: The training data.
        valid_dataloader: The validation data.
        criterion: The loss function.
        device: The device to use.
    
    Returns:
        loss_value, accuracy 
    """
    # set model to evaluation mode
    #turn off gradient computation
    model.eval()

    # init 
    loss_value,accuracy = 0,0
    
    #start time 
    t0 = time.time()

    #Iterate over the pairs (X_i,y_i) (features, labels)
    for i, (X,y) in enumerate(dataloader):
        #if i >= 10:
        #    break
        #move the data to device
        #must be on the same device that model
        features,labels = X.to(device),y.to(device)
        
        # apply forward propagation
        # outputs is the log probabilities
        outputs = model(features)

        # compute loss
        loss= criterion(outputs, labels)

        # the loss_value is a number
        batch_loss = loss.item()
        loss_value+=  batch_loss

        # Get the classes predicted by the model
        # logits -> classes 
        predicted_classes = th.argmax(
            th.softmax(outputs,dim = 1),
            dim= 1 
        )
        
        # Compute accuracy
        # log probabilities -> probabilities
        probs = th.exp(outputs)

        # get the topk biggest values from probs
        # values, indices = probs.topk(k=5, dim=1)  
        # we get top probability, top class (index)
        top_p, top_class = probs.topk(k=1, dim=1)

        # compare the predicted class (top_class) and the actual classes (labels)
        # transform (get a view) the labels into the shape of top class
        equals = top_class == labels.view(*top_class.shape)
        #train_accuracy += th.mean(equals.type(th.FloatTensor)).item()
        accuracy += th.mean(equals.type(th.float)).item()
        
    loss_value = loss_value / len(dataloader)
    accuracy = accuracy / len(dataloader)

    print('\033[102m' + 'New validation epoch completed!' + '\033[0m')
    print(f'Validation Loss: {loss_value:.3f}')
    print(f'Validation Accuracy: {100.0*accuracy:.3f}%')
    print(f'Time Elapsed: {time.time() - t0:.3f} seconds')
    print()    

    return loss_value, accuracy



def train(model: th.nn.Module,
          train_dataloader: th.utils.data.DataLoader,
          valid_dataloader: th.utils.data.DataLoader,
          criterion: th.nn.Module,
          optimizer: th.optim.Optimizer,
          device: th.device, 
          epochs : int) -> tuple[list[float],list[float],list[float],list[float]]:
    """
    Receive a model, train it with the given optimizer
    by ONE epoch, and evelate the model.
    using the `criterion` function to measure the performance of the model.
    The model will be trained on the `train_dataloader` and
    evaluated on the `valid_dataloader` during each epoch.
    The function will return the trained model.
    
    Args:
        model: The model to train.
        train_dataloader: The training data.
        valid_dataloader: The validation data.
        criterion: The loss function.
        optimizer: The optimizer to use.
        device: The device to use.
        epochs: the number of epochs to train
    Model weigths are changed in place
    
    Returns:
        train_losses, train_accuracies, valid_losses, valid_accuracies
        each one is a list with the historic losses and accuracies
    """

    # Collect loss and accuracy 
    train_losses, train_accuracies = [],[]
    valid_losses, valid_accuracies = [],[]

    print("\033[92m{}\033[00m".format("Starting training . . . ") )
    print("Number of epochs: " ,epochs)
    print("Device: ",device )
    print("Criterion: ", criterion)
    print("Optimizaer: ", optimizer)

    for epoch in range(epochs):
      print("\033[94m {}\033[00m".format("Epoch: " + str(epoch)))
          
      train_loss, train_accuracy = train_step(model = model ,
        dataloader = train_dataloader,
        criterion = criterion,
        optimizer = optimizer,
        device = device)
      
      train_losses.append(train_loss)
      train_accuracies.append(train_accuracy)
      
      valid_loss, valid_accuracy = validation_step(model = model ,
        dataloader = valid_dataloader,
        criterion = criterion,
        device = device)
      valid_losses.append(valid_loss)
      valid_accuracies.append(valid_accuracy)
      
      print()
    
    return train_losses, train_accuracies, valid_losses, valid_accuracies