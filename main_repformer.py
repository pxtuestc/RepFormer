import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import dataset2016a, dataset2018_01a
import datetime
import argparse
from RepFormer import RepFormer
from utils import *


def main():
    
    parser = argparse.ArgumentParser(description='PyTorch for NRLKNet')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--classes', type=int, default=11, metavar='N',
                        help='number of classes (default: 11)')
    parser.add_argument('--n', type=int, default=2, metavar='N',
                        help='number of Conv-BN branches (default: 2)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1.3e-3, metavar='M',
                        help='weight decay coefficient (default: 2e-3)')
    parser.add_argument('--factor', type=float, default=0.5, metavar='M',
                        help='learning rate attenuation multiplier factor (default: 0.5)')
    parser.add_argument('--patience', type=int, default=5, metavar='M',
                        help='monitor how many epochs the indicator is unchanged and adjust the learning rate (default: 5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--load', action='store_true', default=True,
                        help='If test the model')
    parser.add_argument('--calculate-snr', action='store_true', default=False,
                        help='calculate the accuracy across SNRs')
    parser.add_argument('--dataset', type=str, default='a', 
                        help='Dataset to load: a for RML2016.10A, b for RML2018.01A (default: a)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/RepFormer_2016_depth5_head5_dim30.pth', 
                        help='Path of the pre-trained model to be loaded')
    args = parser.parse_args()
    
    
    if args.dataset.lower() == 'a':
        (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = dataset2016a.load_data()
        
    elif args.dataset.lower() == 'b':
        (mods, snrs, Z_array), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = dataset2018_01a.load_data()
        
    else:
        raise ValueError(f"Dataset error: Allowed values are a, or b, but the input is: {args.dataset}")
    
    X_train=np.expand_dims(X_train,axis=1)     # Add channel
    X_test=np.expand_dims(X_test,axis=1)

    loss_func = nn.CrossEntropyLoss()
    
    # Load Dataset
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    
    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = RepFormer(img_size=(2, 128),
                          patch_size=(2, 16),
                          n_classes=args.classes,
                          depth=5,
                          n_heads=5,
                          embed_dim=30,
                          mlp_ratio=1,
                          qkv_bias=False,
                          init_values=1,
                          pe=False,
                          n_branch=args.n,
                          ).to(device)
    
    print(net)

    # Validate the pre-trained model, we provide the un-reparameterized model for verifying the reparameterization process.
    if args.load:
        net.load_state_dict(torch.load(args.checkpoint))
        net.eval()
        print('Un-reparameterized Model: \n', net)
        print('Test un-reparameterized model...')
        net_test(net, test_load, 0, loss_func, 0, device)  
        
        net.switch_to_deploy()  # Reparameterize the model
        print('Reparameterized Model: \n', net)
        print('Test reparameterized model...')
        net_test(net, test_load, 0, loss_func, 0, device) 
        return
    
    if args.calculate_snr:
        net.load_state_dict(torch.load(args.checkpoint))
        calculate_acc_per_snr(net)
        return
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=0.0000001, verbose=1)  

    start_time = datetime.datetime.now()

    best_acc = 0
    for epoch in range(0, args.epochs):

        net_train(net, train_load, optimizer, epoch, args.log_interval, loss_func, device)

        val_loss, best_acc = net_test(net, test_load, epoch, loss_func, best_acc, device)

        scheduler.step(best_acc)

    end_time = datetime.datetime.now()


if __name__ == '__main__':
    main()
