from data_utils import Dataset_loader
from model import ParallelModel
import torch
from torch import nn
import numpy as np
import os, glob
import subprocess
import argparse
from sklearn.preprocessing import StandardScaler

def make_train_step(model, loss_fnc, optimizer):
    def train_step(X,Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100
    return train_step

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

def save_model(model, epoch):
    SAVE_PATH = os.path.join(os.getcwd(),'trained_model')
    os.makedirs('models',exist_ok=True)
    torch.save(model.state_dict(),os.path.join(SAVE_PATH, f'{epoch}_cnn_transf_parallel_model.pt'))
    print('Model is saved to {}'.format(os.path.join(SAVE_PATH, f'{epoch}_cnn_transf_parallel_model.pt')))
    if len(os.listdir(SAVE_PATH)) > 5:
        del_model(SAVE_PATH)

def del_model(save_path):
    files = glob.glob(os.path.join(save_path, '*')) 
    old = min(files, key=os.path.getmtime)
    old = os.path.join(save_path, old)
    subprocess.check_call(f'rm -rf "{old}"', shell=True)
    print(f'Delete oldest model: {old}')

def reshape_data(data):
    data = np.expand_dims(data, 1)
    scaler = StandardScaler()
    b, c, h, w = data.shape
    data = np.reshape(data, newshape=(b, -1))
    data = scaler.fit_transform(data)
    data = np.reshape(data, newshape=(b, c, h, w))
    return data
    
def main(parser):
    args = parser.parse_args()
    # print(args)
    # sample_rate=16000
    # test_list='/sd0/jelee/datasets/ESD_test_metadata.txt'
    # train_list='/sd0/jelee/datasets/ESD_train_metadata.txt'
    # valid_list='/sd0/jelee/datasets/ESD_validation_metadata.txt'

    print('------------------- Load Data -------------------')

    traindata = Dataset_loader(args.train_list, args)
    valdata = Dataset_loader(args.valid_list, args)
    
    X_train, Y_train = traindata.__getitem__()
    print('Number of train data: {}'.format(len(Y_train)))
    X_val, Y_val = valdata.__getitem__()
    print('Number of valid data: {}'.format(len(Y_val)))
    
    print('--------------- Finish data loading ---------------')
    
    EPOCHS=200
    DATASET_SIZE = X_train.shape[0]
    BATCH_SIZE = 16
    model = ParallelModel(num_emotions=5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # print('Selected device is {}'.format(device))
    model.to(device)
    # model = ParallelModel(num_emotions=len(EMOTIONS)).to(device)
    print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )
    OPTIMIZER = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model,loss_fnc)
    
    print('--------------- Reshape data ---------------')
    X_train = reshape_data(X_train)
    X_val = reshape_data(X_val)
    
    # print(f'X_train: {X_val.shape}, Y_train: {Y_val.shape}')
    print(f'X_train: {X_train.shape}, X_val: {X_val.shape}')
    print(f'Y_train: {Y_train.shape}, Y_val: {Y_val.shape}')
    
    losses=[]
    max_acc = None
    last_epoch = 0
    cnt = 0
    val_losses = []
    print('--------------- Start model training ---------------')
    for epoch in range(EPOCHS):
        # schuffle data
        ind = np.random.permutation(DATASET_SIZE)
        X_train = X_train[ind,:,:,:]
        Y_train = Y_train[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end-batch_start
            X = X_train[batch_start:batch_end,:,:,:]
            Y = Y_train[batch_start:batch_end]
            X_tensor = torch.tensor(X,device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
            loss, acc = train_step(X_tensor,Y_tensor)
            epoch_acc += acc*actual_batch_size/DATASET_SIZE
            epoch_loss += loss*actual_batch_size/DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}",end='')
        X_val_tensor = torch.tensor(X_val,device=device).float()
        Y_val_tensor = torch.tensor(Y_val,dtype=torch.long,device=device)
        val_loss, val_acc, predictions = validate(X_val_tensor,Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print('')
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%", flush=True)
        # validation accuracy 가장 높은 모델 저장
        if (max_acc is None) or (val_acc >= max_acc):
            max_acc = val_acc
            save_model(model, epoch)
            cnt = 0
        else:
            cnt += 1
        
        if cnt > 20:
            last_epoch = epoch
            break
    
    print('The model training was completed at epoch {}'.format(last_epoch))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EmotionClassification")
    parser.add_argument('--train_list', type=str, default="/sd0/jelee/datasets/ESD_train_metadata.txt", help='Train list');
    parser.add_argument('--valid_list', type=str, default="/sd0/jelee/datasets/ESD_validation_metadata.txt", help='Validation list');
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling Rate');
    
    main(parser)