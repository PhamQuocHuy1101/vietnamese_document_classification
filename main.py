# import trainer
# import prediction

import sys
import config

if __name__ == '__main__':
    config.read_config('./config/main.yml')
    mode = sys.argv[1]
    if mode == 'server':
        import server
    elif mode == 'predict':
        test_file_path = sys.argv[2]
        if not test_file_path:
            print('Need test file path')

        print("Loading model...")
        import prediction
        label = prediction.predict_from_file(test_file_path)
        print("Label: ", label)
    elif mode == 'train':
        import trainer
        print("Loading model...")
    else:
        print("Mode {'train', 'train'}")
