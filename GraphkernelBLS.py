import numpy as np
import torch
import torch.nn.functional as F

class GraphkernelBLS:

    def __init__(self, sigma=1, reg_para=1.,n_layer=2, N2=2):
        self.reg_para = reg_para
        self.n_layer = n_layer
        self.sigma = sigma
        self.N2 =N2
        self.H_tr=[]
        self.H_ts=[]
        self.B=[]
        self.y_pre=[]

    def rbf_kernel_X(self, X, gamma):
        n = X.shape[0]
        Sij = torch.matmul(X, X.T)
        Si = torch.unsqueeze(torch.diag(Sij), 0).T @ torch.ones(1, n).to(X.device)
        Sj = torch.ones(n, 1).to(X.device) @ torch.unsqueeze(torch.diag(Sij), 0)
        D2 = Si + Sj - 2 * Sij
        K = torch.exp(-D2 * gamma)
        return K

    def rbf_kernel_K(self, K_t, gamma):
        n = K_t.shape[0]
        s = torch.unsqueeze(torch.diag(K_t), 0)
        D2 = torch.ones(n, 1).to(K_t.device) @ s + s.T @ torch.ones(1, n).to(K_t.device) - 2 * K_t
        K = torch.exp(-D2 * gamma)
        return K


    def fit(self, X, y_tr, A_norm, idx_train, idx_test, device):
        self.sample_weight = None
        label=y_tr[idx_train]
        num_classes = len(torch.unique(y_tr))
        y_tr = F.one_hot(y_tr, num_classes=num_classes)
        y_tr = y_tr[idx_train]
        
        y_pre=[]
  
        for i in range(self.n_layer):
            Z = []
            for _ in range(self.N2):
                X_t = A_norm @ X
                A1 = self.rbf_kernel_X(X_t, self.sigma)
                
                Z.append(A1)
            Z = torch.hstack(Z)
            H1 = A_norm @ Z @ A_norm.t()
        
            Hp = self.rbf_kernel_K(H1, self.sigma)

            H_ = torch.hstack((Z, Hp))

            X=H_
            
            A1 = H_[np.ix_(idx_train, idx_train)]
            self.H_tr.append(A1)
            B1 = H_[np.ix_(idx_test, idx_train)]
            self.H_ts.append(B1)

            y_tr = y_tr.to(device).float()

            if self.H_tr[i].shape[0] >= self.H_tr[i].shape[1]:
                TT = torch.matmul(self.H_tr[i].T, self.H_tr[i]) + self.reg_para * torch.eye(self.H_tr[i].shape[1], device=self.H_tr[i].device)
                inv_ = torch.linalg.inv(TT)
                A = torch.matmul(torch.matmul(inv_, self.H_tr[i].T), y_tr)  # Ensure y_tr is Float
                self.B.append(A)
            else:
                TT = torch.matmul(self.H_tr[i], self.H_tr[i].T) + self.reg_para * torch.eye(self.H_tr[i].shape[0], device=self.H_tr[i].device)
                inv_ = torch.linalg.inv(TT)
                A = torch.matmul(torch.matmul(self.H_tr[i].T, inv_), y_tr.float())  # Ensure y_tr is Float
                self.B.append(A)

        for i in range(self.n_layer):
            H_tr_i = torch.tensor(self.H_tr[i], dtype=torch.float32) if not isinstance(self.H_tr[i], torch.Tensor) else self.H_tr[i]
            B_i = torch.tensor(self.B[i], dtype=torch.float32) if not isinstance(self.B[i], torch.Tensor) else self.B[i]

            y_pre1 = torch.matmul(H_tr_i, B_i)

            y_pre.append(y_pre1)

        vote_res = [torch.argmax(item, dim=1) for item in y_pre]
        vote_res = list(map(torch.bincount, list(torch.stack(vote_res).transpose(0, 1))))
        vote_res = torch.tensor(list(map(torch.argmax, vote_res)))
        vote_res=vote_res.to(device)
        vote_acc = torch.sum(torch.eq(vote_res, label)) / len(label)
        return vote_acc
