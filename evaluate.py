from data_utils import Dataset_loader
from model import ParallelModel
import torch
from torch import nn
import numpy as np
import os, glob
import subprocess
import argparse
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

def make_validate_fnc(model,loss_fnc):
    def validate(X,Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = loss_fnc(output_logits,Y)
        return loss.item(), accuracy*100, predictions
    return validate

def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)

def reshape_data(data):
    data = np.expand_dims(data, 1)
    scaler = StandardScaler()
    b, c, h, w = data.shape
    data = np.reshape(data, newshape=(b, -1))
    data = scaler.fit_transform(data)
    data = np.reshape(data, newshape=(b, c, h, w))
    return data

def find_latest_model(path):
    files = glob.glob(os.path.join(path, '*')) 
    latest = max(files, key=os.path.getmtime)
    return latest
    
def load_state(path):
    loaded_state = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in loaded_state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict
            
def main(parser):
    args = parser.parse_args()
    # print(args)
    # sample_rate=16000
    # test_list='/sd0/jelee/datasets/ESD_test_metadata.txt'
    # train_list='/sd0/jelee/datasets/ESD_train_metadata.txt'
    # valid_list='/sd0/jelee/datasets/ESD_validation_metadata.txt'

    print('| Start loading Data ----')

    testdata = Dataset_loader(args.test_list, args)
    X_test, Y_test = testdata.__getitem__()
    print('| Number of test data: {}'.format(len(Y_test)))

    print('| Finish data loading ---')
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ParallelModel(num_emotions=5).to(device)
    
    print('| Start loading a model state')
    LOAD_PATH = os.path.join(os.getcwd(),'trained_model')
    latest_model = find_latest_model(LOAD_PATH)
    latest_path = os.path.join(LOAD_PATH, latest_model)
    state_dict = load_state(latest_path)
    model.load_state_dict(state_dict)
    print('| Model is loaded from {}'.format(latest_model))
    print()
    
    validate = make_validate_fnc(model, loss_fnc)
    
    print('| Reshape data ---')
    X_test = reshape_data(X_test)
    
    print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')
    print()
    
    X_test_tensor = torch.tensor(X_test,device=device).float()
    Y_test_tensor = torch.tensor(Y_test,dtype=torch.long,device=device)
    test_loss, test_acc, predictions = validate(X_test_tensor,Y_test_tensor)
    
    print(f'| Test loss is {test_loss:.3f}')
    print(f'| Test accuracy is {test_acc:.2f}%')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EmotionClassification")
    parser.add_argument('--test_list', type=str, default="/sd0/jelee/datasets/ESD_test_metadata.txt", help='Test list');
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling Rate');
    
    main(parser)