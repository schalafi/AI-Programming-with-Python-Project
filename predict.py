import argparse 

from predict_utils import Predictor 

def get_input_args():
    """
    Get the args for 
    predictiong on an image
    """

    parser = argparse.ArgumentParser(
        description = 'Pass an image and predict using the model on the checkpoint'

    )
    parser.add_argument('input',
                        type= str,
                        action ='store',
                        help = 'Path to the image to predict on')

    parser.add_argument('checkpoint',
                        type = str,
                        action = 'store',
                        help = "Path to the model's checkpoint  (a .pth file)"
                        )
    parser.add_argument('--top_k',
                        type = int,
                        action = 'store',
                        default = 5,
                        help = 'Get the top k most likely classes')
    parser.add_argument('--category_names',
                        type = str,
                        action = 'store',
                        default='cat_to_name.json',
                        help = 'Path to the json file of the  mapping of category (int) to name (string)\n default is cat_to_name.json')
    parser.add_argument('--gpu',
                        action = 'store_true',
                         help = 'Whether to use GPU for inference. if you pass --gpu it will use gpu while predicting')

    return parser     


if __name__ == "__main__":
    
    # get the input's parser
    parser = get_input_args()
    args = parser.parse_args()

    print("Input args are: \n", args)


    # test with 
    # python predict.py sunflowers.webp ./models/20231006-034221-checkpoint.pth --category_names cat_to_name.json --gpu

    predictor = Predictor(args)

    predictor.predict()
