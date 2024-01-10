import torch
import os
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset

# class Dataset(torch.uitls.data.Dataset):
#     def __init__(self, args):
#         self.labels = open(args.input_dir, 'r').readlines()
#         self.emo_dict_rev = {'Happy': 0, 'Sad': 1, 'Angry': 2, 'Surprise': 3}
#         self.emo_dict = {0:'Happy', 1:'Sad', 2:'Angry', 3:'Surprise'}
#         self.data = pd.DataFrame(columns=['Emotion', 'Path'])
#         self.emo_lst = []
#         self.path_lst = []
#         self.sample_rate = 16000
#         self.mel_spectrograms = []
#         self.signals = []
        
#     def __getitem__(self):
#         # Load label datas
#         for label in self.labels:
#             _, _, emotion, _, wav_fn = label.split('/t')
#             emo_idx = self.emo_dict_rev[emotion]
#             self.emo_lst.append(emo_idx)
#             self.path_lst.append(wav_fn)
#             # new_row = pd.DataFrame([[emo_idx, wav_fn]], columns=['Emotion', 'Path'])
#             # self.data = pd.concat([self.data, new_row], ignore_index=True)
#         print("number of files is {}".format(len(self.emo_lst)))
        
#         # Load the signals
#         for i, file_path in enumerate(self.path_lst):
#             audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=self.sample_rate)
#             signal = np.zeros((int(self.sample_rate*3,)))
#             signal[:len(audio)] = audio
#             self.signals.append(signal)
#             print("\r Processed {}/{} files".format(i,len(self.path_lst)),end='')
#         self.signals = np.stack(self.signals,axis=0)
        
#         # Split the data
#         X = self.signals
#         Y = np.array(self.emo_lst)
#         # X_train,X_val,X_test = [],[],[]
#         # Y_train,Y_val,Y_test = [],[],[]
        
#         X_test, X_val, X_train = X[:1500, :], X[1500:2500, :], X[2500:, :]
#         Y_test, Y_val, Y_train = Y[:1500, :], Y[1500:2500, :], Y[2500:, :]
        
#         X_train = np.concatenate(X_train,0)
#         X_val = np.concatenate(X_val,0)
#         X_test = np.concatenate(X_test,0)
#         Y_train = np.concatenate(Y_train,0)
#         Y_val = np.concatenate(Y_val,0)
#         Y_test = np.concatenate(Y_test,0)
        
#         # print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
#         # print(f'X_val:{X_val.shape}, Y_val:{Y_val.shape}')
#         # print(f'X_test:{X_test.shape}, Y_test:{Y_test.shape}')
        
#         # Get mel-spectrogram
#         mel_train = []
#         print("Calculatin mel spectrograms for train set")
#         for i in range(X_train.shape[0]):
#             mel_spectrogram = getMELspectrogram(X_train[i,:], sample_rate=self.sample_rate)
#             mel_train.append(mel_spectrogram)
#             print("\r Processed {}/{} files".format(i,X_train.shape[0]),end='')
#         print('')
#         mel_train = np.stack(mel_train,axis=0)
#         del X_train
#         X_train = mel_train

#         mel_val = []
#         print("Calculatin mel spectrograms for val set")
#         for i in range(X_val.shape[0]):
#             mel_spectrogram = getMELspectrogram(X_val[i,:], sample_rate=self.sample_rate)
#             mel_val.append(mel_spectrogram)
#             print("\r Processed {}/{} files".format(i,X_val.shape[0]),end='')
#         print('')
#         mel_val = np.stack(mel_val,axis=0)
#         del X_val
#         X_val = mel_val

#         mel_test = []
#         print("Calculatin mel spectrograms for test set")
#         for i in range(X_test.shape[0]):
#             mel_spectrogram = getMELspectrogram(X_test[i,:], sample_rate=self.sample_rate)
#             mel_test.append(mel_spectrogram)
#             print("\r Processed {}/{} files".format(i,X_test.shape[0]),end='')
#         print('')
#         mel_test = np.stack(mel_test,axis=0)
#         del X_test
#         X_test = mel_test   
    
#         return X_test, X_val, X_train, Y_test, Y_val, Y_train
        
def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                            sr=sample_rate,
                                            n_fft=1024,
                                            win_length = 512,
                                            window='hamming',
                                            hop_length = 256,
                                            n_mels=128,
                                            fmax=sample_rate/2
                                            )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
        
            
            
class Dataset_loader(Dataset):
    def __init__(self, data_list, args):

        self.data_list = data_list
        self.sample_rate = args.sample_rate
        
        # Read training files
        with open(data_list) as dataset_file:
            lines = dataset_file.readlines();
        # print('lines: ', lines)

        # Make a dictionary of emotion ID and ID indices
        dictkeys = list(set([x.split('\t')[2] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split('\t')

            emotion_label = dictkeys[data[2]]
            filename = data[-1]
            
            self.data_label.append(emotion_label)
            self.data_list.append(filename)
        
        self.data_label = np.array(self.data_label, dtype=np.int32)

    def __getitem__(self):
        feat = []
        label = []
        exp_cnt = 0
        # print('data_label: ', len(self.data_label), self.data_label)
        for index in range(len(self.data_label)):
            # print('data_file: ', self.data_list[index])
            # print('emotion_label: ', self.data_label[index])
            try:
                audio, sr = librosa.load(self.data_list[index], duration=3, offset=0.5, sr=self.sample_rate)
                signal = np.zeros((int(self.sample_rate*3,)))
                signal[:len(audio)] = audio
                mel = getMELspectrogram(signal, self.sample_rate)
                feat.append(mel);
                label.append(np.array([self.data_label[index]], dtype=np.int32))
            except FileNotFoundError:
                print(f'FileNotFoundError: No file {self.data_list[index]}')
                exp_cnt += 1

        feat = np.stack(feat, axis=0)      # feat.shape = (1000, 128, 188)
        label = np.concatenate(label, axis=0)   # label.shape = (1000,)
        print('| Number of Exception file: {}'.format(exp_cnt))
        # print(feat.shape)
        # print(label.shape)
        return torch.FloatTensor(feat), label

    def __len__(self):
        return len(self.data_list)