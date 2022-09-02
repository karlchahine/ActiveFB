
#################################################################################
#######  c is of length 1  ########
#################################################################################
##this is the one

#################################
#######  Libraries Used  ########
#################################
import os
import sys
import argparse
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

#################################
#######  Parameters  ########
#################################
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-init_nw_weight', type=str, default='default')
    parser.add_argument('-code_rate', type=int, default=3)
    parser.add_argument('-precompute_stats', type=bool, default=True)  ##########

    parser.add_argument('-learning_rate', type = float, default=0.0001)
    parser.add_argument('-batch_size', type=int, default=500)
    parser.add_argument('-num_epoch', type=int, default=5000)

    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')

    parser.add_argument('-block_len', type=int, default=2)
    parser.add_argument('-num_block', type=int, default=50000)


    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=2)
    parser.add_argument('-fb_num_layer',  type=int, default=2)
    parser.add_argument('-enc_num_unit',  type=int, default=50)
    parser.add_argument('-dec_num_unit',  type=int, default=50)
    parser.add_argument('-fb_num_unit',   type=int, default=50)

    parser.add_argument('-frwd_snr', type=float, default= 0)
    parser.add_argument('-bckwd_snr', type=float, default= 0)

    parser.add_argument('-snr_test_start', type=float, default=0.0)
    parser.add_argument('-snr_test_end', type=float, default=5.0)
    parser.add_argument('-snr_points', type=int, default=6)

    parser.add_argument('-channel_mode', choices=['normalize', 'lazy_normalize', 'tanh'], default='lazy_normalize')

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'none'], default='elu')
    parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'none'], default='none')



    args = parser.parse_args()

    return args

class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args             = args


        # Encoder
        self.enc_rnn_fwd   = torch.nn.GRU(args.block_len + args.block_len//2, self.args.enc_num_unit,
                                           num_layers=self.args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) 

        self.enc_linear    = torch.nn.Linear(self.args.enc_num_unit, args.block_len//2)



        # Decoder
        self.dec_rnn_1           = torch.nn.GRU(args.block_len//2, self.args.dec_num_unit,
                                           num_layers=self.args.dec_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False)

        self.dec_output_1        = torch.nn.Linear(self.args.dec_num_unit, self.args.block_len)
        
        self.dec_rnn_2           = torch.nn.GRU(args.block_len, self.args.dec_num_unit,
                                           num_layers=self.args.dec_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False)

        self.dec_output_2        = torch.nn.Linear(self.args.dec_num_unit, args.block_len//2)

    ##Power Constraint
    def power_constraint(self, inputs):
        this_mean = torch.mean(inputs)
        this_std  = torch.std(inputs)
        outputs   = (inputs - this_mean)*1.0/this_std
        return outputs

    ##Encoder Activation
    def enc_act(self, inputs):
        if self.args.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.enc_act == 'elu':
            return F.elu(inputs)
        elif self.args.enc_act == 'relu':
            return F.relu(inputs)
        elif self.args.enc_act == 'selu':
            return F.selu(inputs)
        elif self.args.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        else:
            return inputs

    ##Decoder Activation
    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        else:
            return inputs


    def forward(self, input, fwd_noise, fb_noise, eval=False,precomp=False, use_precomp=False,mu_enc=[0.0,0.0,0.0,0.0,0.0,0.0],v_enc=[0.0,0.0,0.0,0.0,0.0,0.0],mu_dec=[0.0,0.0,0.0,0.0,0.0],v_dec=[0.0,0.0,0.0,0.0,0.0]):
        enc=[]
        dec=[]



        enc_input        = torch.cat([input.view(self.args.batch_size, 1, self.args.block_len),
                                            torch.zeros((self.args.batch_size, 1, self.args.block_len//2)).to(device)+0.5], dim=2) 

        enc_output, enc_state  = self.enc_rnn_fwd(enc_input)
        enc_output         = self.enc_act(self.enc_linear(enc_output)) 
        enc.append(enc_output)
        if use_precomp:
            enc_output   = (enc_output - mu_enc[0])*1.0/v_enc[0]

        else:
            enc_output  = self.power_constraint(enc_output)

        dec_input = enc_output  + fwd_noise[:,:,:,0].view(self.args.batch_size, 1, self.args.block_len//2)
        dec_output, dec_state_1  = self.dec_rnn_1(dec_input)
        dec_output = self.dec_act(self.dec_output_1(dec_output))

        dec_output, dec_state_2  = self.dec_rnn_2(dec_output)
        dec_output = self.dec_act(self.dec_output_2(dec_output))
        dec.append(dec_output)
        if use_precomp:
            dec_output   = (dec_output - mu_dec[0])*1.0/v_dec[0]

        else:
            dec_output = self.power_constraint(dec_output)

        noisy_dec_output = dec_output + fb_noise[:,:,:,0].view(self.args.batch_size, 1, self.args.block_len//2)

  
        num_iter = 6
        for time_step in range (1 , num_iter):

            enc_input        = torch.cat([input.view(self.args.batch_size, 1, self.args.block_len),
                                            noisy_dec_output.view(self.args.batch_size, 1, self.args.block_len//2)], dim=2)
                

            enc_output, enc_state  = self.enc_rnn_fwd(enc_input, enc_state)
            enc_output         = self.enc_act(self.enc_linear(enc_output))
            enc.append(enc_output)
            if use_precomp:
                enc_output   = (enc_output - mu_enc[time_step])*1.0/v_enc[time_step]

            else:
                enc_output  = self.power_constraint(enc_output)

            dec_input = enc_output  + fwd_noise[:,:,:,time_step].view(self.args.batch_size, 1, self.args.block_len//2)
            dec_output, dec_state_1  = self.dec_rnn_1(dec_input, dec_state_1)

            if time_step == num_iter-1:
                final_result = F.sigmoid(self.dec_output_1(dec_output))

                if precomp:
                    return enc,dec

                return final_result

            else:
                dec_output = self.dec_act(self.dec_output_1(dec_output))
                dec_output, dec_state_2  = self.dec_rnn_2(dec_output, dec_state_2)
                dec_output = self.dec_act(self.dec_output_2(dec_output))
                dec.append(dec_output)
                if use_precomp:
                    dec_output   = (dec_output - mu_dec[time_step])*1.0/v_dec[time_step]
                else:
                    dec_output = self.power_constraint(dec_output)
                noisy_dec_output = dec_output + fb_noise[:,:,:,time_step].view(self.args.batch_size, 1, self.args.block_len//2)

        


###### MAIN
args = get_args()
print (args)

def errors_ber(y_true, y_pred):


    t1 = np.round(y_true[:,:,:])
    t2 = np.round(y_pred[:,:,:])

    myOtherTensor = np.not_equal(t1, t2).float()
    k = sum(sum(sum(myOtherTensor)))/(myOtherTensor.shape[0]*myOtherTensor.shape[1]*myOtherTensor.shape[2])
    return k

#use_cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    model = AE(args).to(device)
else:
    model = AE(args)

print (model)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

test_ratio = 1
num_train_block, num_test_block = args.num_block, args.num_block/test_ratio

frwd_snr = args.frwd_snr
frwd_sigma = 10**(-frwd_snr*1.0/20)

bckwd_snr = args.bckwd_snr
bckwd_sigma = 10**(-bckwd_snr*1.0/20)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(int(num_train_block/args.batch_size)):

        X_train    = torch.randint(0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float)
        fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len//2, 6), dtype=torch.float)
        fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len//2, 6), dtype=torch.float)

        # use GPU
        X_train, fwd_noise, fb_noise = X_train.to(device), fwd_noise.to(device), fb_noise.to(device)

        optimizer.zero_grad()
        output = model(X_train, fwd_noise, fb_noise)

        loss = F.binary_cross_entropy(output, X_train)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        #if batch_idx % 1000 == 0:
        #print('Train Epoch: {} [{}/{} Loss: {:.6f}'.format(
        #    epoch, batch_idx, num_train_block/args.batch_size, loss.item()))

    # print(output[0,:,:])
    # print(X_train[0,:,:])
    print('====> Epoch: {}, Average BCE loss: {:.4f}'.format(epoch, train_loss /(num_train_block/args.batch_size)))
    


def test_2(model):
    model.eval()
    #torch.manual_seed(random.randint(0,1000))

    frwd_snr = args.frwd_snr
    frwd_sigma = 10**(-frwd_snr*1.0/20)

    bckwd_snr = args.bckwd_snr
    bckwd_sigma = 10**(-bckwd_snr*1.0/20)

    
    test_ber=.0

    with torch.no_grad():
        mu_enc=[0.0,0.0,0.0,0.0,0.0,0.0]
        v_enc=[0.0,0.0,0.0,0.0,0.0,0.0]

        mu_dec=[0.0,0.0,0.0,0.0,0.0]
        v_dec=[0.0,0.0,0.0,0.0,0.0]

        if args.precompute_stats:
            ##Step 1: save mean and var
            num_test_batch = 5000
            for batch_idx in range(num_test_batch):
                X_test     = torch.randint(0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float)
                fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len//2, 6), dtype=torch.float)
                fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len//2, 6), dtype=torch.float)

                # use GPU
                X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

                enc,dec= model(X_test, fwd_noise, fb_noise, False, True, False)

                for i in range(6):
                    mu_enc[i] += torch.mean(enc[i])
                    v_enc[i] += torch.std(enc[i])
                for i in range(5):
                    mu_dec[i] += torch.mean(dec[i])
                    v_dec[i] += torch.std(dec[i])

            for i in range(6):
                mu_enc[i] /= num_test_batch
                v_enc[i] /= num_test_batch
            for i in range(5):
                mu_dec[i] /= num_test_batch
                v_dec[i] /= num_test_batch




        ##Step 2: compute ber
        num_test_batch = 20000
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float)
            fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len//2, 6), dtype=torch.float)
            fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len//2, 6), dtype=torch.float)

            # use GPU
            X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

            X_hat_test = model(X_test, fwd_noise, fb_noise, False, False, args.precompute_stats, mu_enc,v_enc,mu_dec,v_dec)
            
            test_ber  += errors_ber(X_hat_test.cpu(),X_test.cpu())
            

    test_ber  /= 1.0*num_test_batch
    print('Test SNR',frwd_snr ,'with ber ', float(test_ber))



train_model=False 
if(train_model==True):

    for epoch in range(1, args.num_epoch + 1):
        train(epoch)
        if epoch%50 == 0:
            torch.save(model.state_dict(), "./final_models_bl2/fb_8.pt")
            #print("Model is saved")
            test_2(model)

else:
    pretrained_model = torch.load("./final_models_bl2/fb_0.pt")
    model.load_state_dict(pretrained_model)
    model.args = args
    test_2(model)



   