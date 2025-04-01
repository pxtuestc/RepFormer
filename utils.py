import torch
import datetime


def calculate_acc_per_snr(net, save_path='acc_per_snr.csv'):
    net.eval()
    lbl_array = np.vstack(lbl)
    lbl_test = lbl_array[test_idx]    
    acc = []
    for snr in snrs:
        indices = np.where(lbl_test[:,1] == str(snr))[0]    
        snr_test = X_test[indices] 
        snr_test = torch.from_numpy(snr_test)
        snr_labels = Y_test[indices]    
        snr_labels = torch.from_numpy(snr_labels)
        with torch.no_grad():
            snr_predictions = net(snr_test.cuda())   

        _, pre_ok = torch.max(snr_predictions.data, 1)
        pre_ok_num = (pre_ok.to('cpu') == snr_labels).sum()
        acc_snr = pre_ok_num * 100. / len(indices)
        acc.append(acc_snr.numpy())
        
        print(f'Accuracy at SNR {snr}: {acc_snr}')

    # np.savetxt(save_path, acc, fmt='%0.6f')
    

def net_train(net, train_data_load, optimizer, epoch, log_interval, loss_func, device):
    net.train()
    begin = datetime.datetime.now()

    total = len(train_data_load.dataset)
    train_loss = 0
    ok = 0

    for i, data in enumerate(train_data_load, 0):  
        signal, label = data
        signal, label = signal.to(device), label.to(device)

        optimizer.zero_grad()
        outs = net(signal)

        loss = loss_func(outs, label.long())
       
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outs.data, 1)
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:   
            loss_mean = train_loss / (i + 1)
            traind_total = (i + 1) * len(label)
            acc = 100. * ok / traind_total
            progress = 100. * traind_total / total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, traind_total, total, progress, loss_mean, acc))

    end = datetime.datetime.now()
    print('One Epoch Spend: ', end - begin)


def net_test(net, test_data_load, epoch, loss_func, best_acc, device):
    net.eval()

    ok = 0
    val_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_data_load):
            signal, label = data
            signal, label = signal.to(device), label.to(device)

            outs = net(signal)
            loss = loss_func(outs, label.long())
            val_loss += loss.item()
            _, pre = torch.max(outs.data, 1)
            ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_load.dataset))
    loss_mean = val_loss / (i + 1)
    
    if acc > best_acc:
        best_acc = acc

    print('Epoch:{}, Loss:{}, Acc:{}, Best Acc:{}\n'.format(epoch, loss_mean, acc, best_acc))

    return loss_mean, best_acc


