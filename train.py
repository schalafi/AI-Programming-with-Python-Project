import argparse 

def get_input_args():
    """
    Get the args for training the network
    """
    parser = argparse.ArgumentParser(
        description = "Train an image classification model"
    )
    parser.add_argument('data_dir',
                        type=str,
                        action='store',
                        help='path to the training, validation and testing images directory')

    parser.add_argument('--save_dir',
                        type=str,
                        action = 'store',
                        help='directory where the checkpoints will be saved')

    parser.add_argument('--arch',
                        type = str,
                        action = 'store',
                        help = 'the architecture of the neural network') 

    parser.add_argument('--learning_rate',
                        type = float,
                        action = 'store',
                        help = 'the learning rate for training the network',
                        default=0.01)

    parser.add_argument('--hidden_units',
                        type = int,
                        action = 'store',
                        help = 'number of hidden units in classifier',
                        #default=512
                        )

    parser.add_argument('--epochs',
                        type = int,
                        action='store',
                        help = 'number of training epochs (one epoch train the network on all the dataset)',
                        #default=5
                        )
    parser.add_argument('--gpu',
                        #type = bool,
                        action = 'store_true',
                        help = 'pass if you want to use gpu, otherwise will not train on gpu',
                        )
    return parser

# Test with 
#python train.py './data/' --save_dir './models'  --arch 'vgg13' --learning_rate 0.03 --hidden_units 1024 --epochs 10 --gpu False

def tests():
    parser = get_input_args()
    args = parser.parse_args([
        './data/',
        '--save_dir',
        './models/',
        '--arch',
        'vgg13',
        '--learning_rate',
        '0.0001',
        '--hidden_units',
        '256',
        '--epochs',
        '20',
        '--gpu']) #use gpu
    print(
        args
        )
    print('Getting args: ')
    print(args.data_dir)
    print(args.save_dir)
    print(args.arch)
    print(args.learning_rate)
    print(args.hidden_units)
    print(args.epochs)
    print(args.gpu)
    print('Done')




    
if __name__ == '__main__':

    #parser = get_input_args()
    #args = parser.parse_args()
    #print all input args
    #print(args)

    print("Tests: ")
    tests()

    
    

