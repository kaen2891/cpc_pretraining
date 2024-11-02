# Cell
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


# Cell
class CPCEncoder(nn.Sequential):
    'CPC Encoder'
    def __init__(self, input_channels, encoder_output_dim, mode):
        super().__init__()
        self.mode = mode
        if self.mode == 'linear':
            self.encoder = nn.Linear(input_channels, encoder_output_dim)
        elif self.mode == '1dcnn':
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),  # 768 -> 384
                nn.ReLU(),
                nn.Conv1d(input_channels, encoder_output_dim, kernel_size=3, stride=1, padding=1),  # 384 -> 192
                nn.ReLU(),
                nn.Conv1d(encoder_output_dim, encoder_output_dim, kernel_size=3, stride=1, padding=1),   # 192 -> 96
            )
        elif self.mode == '2dcnn':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),  # 768 -> 384
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 384 -> 192
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 192 -> 96
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 96 -> 48
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 48 -> 24
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 24 -> 12
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(2, 1), stride=(2, 1), padding=(2, 0))  # 12 -> 6 (target dimension)
            )
        
    def forward(self, x):        
        print('input x', x.size())
        # x: B, D, T
        
        if self.mode == 'linear':
            x = x.squeeze(1) ## check
            x = x.transpose(1, 2) # B, T, D
            x = self.encoder(x)
        elif self.mode == '1dcnn':
            x = x.squeeze(1) ## check
            x = self.encoder(x)
            x = x.transpose(1, 2) # B, T, D
        
        else:
            x = x.unsqueeze(1)
            x = self.encoder(x)
            x = x.view(x.size(0), x.size(-1), x.size(1) * x.size(2)) # B, T, D
        return x
        

class CPCModel(nn.Module):
    "CPC model"
    def __init__(self, input_channels, encoder_output_dim, mode=None, n_hidden=512, n_layers=2, mlp=False, lstm=True, num_classes=None):
        super().__init__()
        self.encoder = CPCEncoder(input_channels, encoder_output_dim, mode) if mode is not None else None
        self.encoder_output_dim = encoder_output_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.mlp = mlp
        
        self.num_classes = num_classes        
        self.rnn = nn.LSTM(self.encoder_output_dim, n_hidden, num_layers=n_layers, batch_first=True) if lstm is True else nn.GRU(self.encoder_output_dim, n_hidden, num_layers=n_layers, batch_first=True)
        if mode is None:
            self.lin = nn.Linear(input_channels, self.n_hidden)
        
        if(num_classes is None): #pretraining
            if(mlp):# additional hidden layer as in simclr
                self.proj = nn.Sequential(nn.Linear(n_hidden, n_hidden),nn.ReLU(inplace=True),nn.Linear(n_hidden, self.encoder_output_dim))
            else:
                self.proj = nn.Linear(n_hidden, self.encoder_output_dim)
                    

    def forward(self, x):
        # input shape [B, D, T]
        if(self.encoder is not None):
            input_encoded = self.encoder(x)
        else:
            input_encoded = x.transpose(1,2) #bs, seq, channels
            input_encoded = self.lin(input_encoded)
        output_rnn, _ = self.rnn(input_encoded) #output_rnn: bs, seq, n_hidden
        if(self.num_classes is None):#pretraining
            return input_encoded, self.proj(output_rnn)
            
    def cpc_loss(self, x, target=None, steps_predicted=5, n_false_negatives=9, negatives_from_same_seq_only=False, eval_acc=False):
        assert(self.num_classes is None)

        input_encoded, output = self.forward(x) #input_encoded: bs, seq, features; output: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1,input_encoded.size(2)) #for negatives below: -1, features
        #print(f"{input_encoded_flat.size()=}")
        
        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]
        
        loss = torch.tensor(0,dtype=torch.float32).to(x.device)
        tp_cnt = torch.tensor(0,dtype=torch.int64).to(x.device)
        
        for i in range(input_encoded.size()[1]-steps_predicted):
            #print(f"{input_encoded[:,i+steps_predicted].size()=}")
            positives = input_encoded[:,i+steps_predicted].unsqueeze(1) #bs,1,encoder_output_dim
            #print(f"{positives.size()=}")
            #print(f"{negatives_from_same_seq_only=}")
            if(negatives_from_same_seq_only): #True
                # seq = 1000
                # n_false_negatives = 128
                idxs = torch.randint(0,(seq-1),(bs*n_false_negatives,)).to(x.device)
            else:#negative from everywhere
                idxs = torch.randint(0,bs*(seq-1),(bs*n_false_negatives,)).to(x.device)
            idxs_seq = torch.remainder(idxs,seq-1) #bs*false_neg
            idxs_seq2 = idxs_seq * (idxs_seq<(i+steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+steps_predicted)).long()#bs*false_neg
            if(negatives_from_same_seq_only):
                idxs_batch = torch.arange(0,bs).repeat_interleave(n_false_negatives).to(x.device)
            else:
                idxs_batch = idxs//(seq-1)
            idxs2_flat = idxs_batch*seq+idxs_seq2 #for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity
            #print(f"{idxs2_flat=}")
            #print(f"{idxs2_flat.size()=}")
            
            
            negatives = input_encoded_flat[idxs2_flat].view(bs,n_false_negatives,-1) #bs*false_neg, encoder_output_dim
            #print(f"{negatives.size()=}")
            candidates = torch.cat([positives,negatives],dim=1)#bs,false_neg+1,encoder_output_dim
            #print(f"{candidates.size()=}")
            #print(f"{output[:,i].size()=}") # (bs, features)
            #print(f"{output[:,i].unsqueeze(1).size()=}")# (bs, 1, features)
            #print(f"{(output[:,i].unsqueeze(1)*candidates).size()=}") # (bs, (1+false_neg), features)
            
            preds=torch.sum(output[:,i].unsqueeze(1)*candidates,dim=-1) #bs,(false_neg+1)
            #print(f"{preds=}")
            #print(f"{preds.size()=}")
            
            targs = torch.zeros(bs, dtype=torch.int64).to(x.device)
            #print(f"{targs=}")
            #print(f"{targs.size()=}")
            
            if(eval_acc):
                preds_argmax = torch.argmax(preds,dim=-1)
                tp_cnt += torch.sum(preds_argmax == targs)
               
            loss += F.cross_entropy(preds,targs)
            #print(f"{loss=}")
        if(eval_acc):
            return loss, tp_cnt.float()/bs/(input_encoded.size()[1]-steps_predicted)
        else:
            return loss


def main():
    #test
    model = CPCModel(input_channels=768, encoder_output_dim=512, mode='2dcnn').cuda()
    data_input = torch.rand(4, 768, 320).cuda() # batch, dim, time
    output, _ = model(data_input)
    print(output.size())
    
    loss, acc = model.cpc_loss(data_input, steps_predicted=12, n_false_negatives=128, negatives_from_same_seq_only=True, eval_acc=True)
    print('loss', loss)
    print('acc', acc)
    

if __name__ == '__main__':
    main()


