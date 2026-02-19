import warnings
warnings.filterwarnings("ignore")
import torch
from Dataloader import load_data
from GraphkernelBLS import GraphkernelBLS

if __name__ == '__main__':
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    # Load data
    path = "./data/"
    dataset = "ACM"
    rand_split = True
    adj, features, labels, n_class, train_idx, idx_val, test_idx = load_data(path, dataset, rand_split, device)

    y=labels.to(device)

    X = features
    
    REG_PARAM = 1e-4
    run_times = 5
    N2 = 2
    acc_list = []

    c_1 = 1e-4
    sigma = 1
    num_layers = 10

    for i in range(run_times):
        REG_PARAM=1/c_1
        base_model = GraphkernelBLS( sigma, reg_para=REG_PARAM, n_layer= num_layers, N2=N2)
        acc_x = base_model.fit(X, y, adj, train_idx, test_idx, device)
        
        print('%s ==> ACC: %s' % (i, acc_x))
        acc_list.append(acc_x)
        acc_tensor = torch.tensor(acc_list, dtype=torch.float32)  
        Acc = acc_tensor.mean().item()  
        std = acc_tensor.std().item() 

    print('MEAN ACC: %s +- %s' % (Acc, std))

                