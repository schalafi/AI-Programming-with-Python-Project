import argparse 

from train_utils import Trainer

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
                        action = 'store_true',
                        help = 'pass if you want to use gpu, otherwise will not train on gpu',
                        )
    return parser


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


def main()-> None:
    """
    Main function for training the network
    Usage examples:
    #vgg16
    python train.py './flowers/' --save_dir './models'  --arch 'vgg16' --learning_rate 0.03 --hidden_units 1024 --epochs 10 --gpu
    #densenet121
    python train.py './flowers/' --save_dir './models'  --arch 'densenet121' --learning_rate 0.03 --hidden_units 1024 --epochs 10 --gpu
    #alexnet
    python train.py './flowers/' --save_dir './models'  --arch 'alexnet' --learning_rate 0.03 --hidden_units 1024 --epochs 10 --gpu
    """
    parser = get_input_args()
    args = parser.parse_args()

    print("Input args are: \n", args)

    trainer = Trainer(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        gpu=args.gpu,
        #for testing purposes
        #n_minibatches = 2
    )

if __name__ == '__main__':
    main()
    

    
    