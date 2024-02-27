import argparse
from preprocess import preprocessing
from add_features import add_features_3, add_features_1, add_features_2

from model_class import train_model as train_model_class, predict_model as predict_model_class
from model_iden import train_model as train_model_iden, predict_model as predict_model_iden

parser = argparse.ArgumentParser(
                    prog='SRL-Tagger',
                    description='A tool for extracting semantic roles from text')


parser.add_argument('op_mode', help='Model or preprocess', choices=['preproc', 'model'])
parser.add_argument('-i', '--input', help='Input file (should be on the corresponding folder according to documentation)', required=True)
parser.add_argument('-mf', '--model_file', help='Model file to use for prediction')

parser.add_argument('-m', '--mode', help='The operation mode of the program', choices=['train', 'predict'])
parser.add_argument('-t', '--task', help='The task to solve', choices=['identify', 'classify'])


if __name__ == '__main__':
    args = parser.parse_args()

    if args.op_mode == 'preproc':
        file_name = args.input.split('.')[0]
        df = preprocessing(args.input) 
        df = add_features_1(df)
        #df = add_features_2(df)
        #df = add_features_3(df)
        df.to_csv(f'data/input/{file_name}_final.tsv', index=False, sep='\t')
    
    elif args.op_mode == 'model':
        if not args.mode:
            raise Exception('Mode is required')
        if not args.task:
            raise Exception('Task is required')
        if args.mode == 'train':
            if args.task == 'identify':
                train_model_iden(args.input)     
            else:
                train_model_class(args.input)
        elif args.mode == 'predict':
            if args.task == 'identify':
                predict_model_iden(args.input, args.model_file)
            else:
                predict_model_class(args.input, args.model_file)
    else:
        raise Exception('Invalid operation mode')