import h5py
import numpy as np


def load_data(dataset_path = 'datasets/GOLD_XYZ_OSC.0001_1024.hdf5'):
    
    h5file = h5py.File(dataset_path, 'r+')

    X = h5file['X'][:]  # 数据 (2555904, 1024, 2)
    
    Y = h5file['Y'][:]  # one-hot标签 (2555904, 24)
    
    Z = h5file['Z'][:]  # SNR (2555904, 1)

    X_array = np.array(X) #IQ
    Y_array = np.array(Y) #mod
    Z_array = np.array(Z) #SNR
    
    mods = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM',  '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC','FM', 'GMSK', 'OQPSK']
    snrs  = range(-20,31,2)
    # x = []
    # lbl = []
    train_idx = []
    test_idx = []
    np.random.seed(2018)
    n = 0
    for i in range(0,24,1):   # 24 MODS
        # os.mkdir(os.path.join(train_path, mod[i]))
        for j in range(0,26,1):    # 26 SNRS
            train_idx += list(np.random.choice(range(n * 4096, (n + 1) * 4096), size=int(np.ceil(4096*0.8)), replace=False))
            n = n + 1

    test_idx = list(set(range(2555904)) - set(train_idx))
    X_train = X_array[train_idx].swapaxes(2,1)       
    Y_train = np.argmax(Y_array[train_idx], axis=1)

    X_test = X_array[test_idx].swapaxes(2,1)
    Y_test = np.argmax(Y_array[test_idx], axis=1)

    return (mods, snrs, Z_array), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx)
 
if __name__ == '__main__':
    load_data()
