##Power Norm where we fix the mean and var
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
    parser.add_argument('-num_epoch', type=int, default=600)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('-block_len', type=int, default=1)
    parser.add_argument('-num_block', type=int, default=50000)
    parser.add_argument('-num_rounds', type=int, default=3)
    #parser.add_argument('-delta_snr', type=int, default=15)  ##SNR_FB - SNR_FW


    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=2)
    parser.add_argument('-enc_num_unit',  type=int, default=50)
    parser.add_argument('-dec_num_unit',  type=int, default=50)

    parser.add_argument('-frwd_snr', type=float, default= 0)
    parser.add_argument('-bckwd_snr', type=float, default= 16)

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

        ##Power Norm weights
        ##Enc
        self.p1_enc = torch.nn.Parameter(torch.randn(()))
        self.p2_enc = torch.nn.Parameter(torch.randn(()))
        self.p3_enc = torch.nn.Parameter(torch.randn(()))

        ##Dec
        self.p1_dec = torch.nn.Parameter(torch.randn(()))
        self.p2_dec = torch.nn.Parameter(torch.randn(()))

        # Encoder
        self.enc_rnn_fwd   = torch.nn.GRU(3*self.args.block_len, self.args.enc_num_unit,
                                           num_layers=self.args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) 

        self.enc_linear    = torch.nn.Linear(self.args.enc_num_unit, int((self.args.code_rate)*(self.args.block_len)/(self.args.num_rounds)))



        # Decoder
        self.dec_rnn           = torch.nn.GRU(int((self.args.code_rate)*(self.args.block_len)/(self.args.num_rounds)), self.args.dec_num_unit,
                                           num_layers=self.args.dec_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False)

        self.dec_output        = torch.nn.Linear(self.args.dec_num_unit, self.args.block_len)
        

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


    def forward(self, input, fwd_noise, fb_noise, eval=False, precomp=False, use_precomp=False,mu_1_enc=0.0,v_1_enc=0.0,mu_2_enc=0.0,v_2_enc=0.0,mu_3_enc=0.0,v_3_enc=0.0,mu_1_dec=0.0,v_1_dec=0.0,mu_2_dec=0.0,v_2_dec=0.0):

        den_enc = torch.sqrt(self.p1_enc**2 + self.p2_enc**2 + self.p3_enc**2)
        den_dec = torch.sqrt(self.p1_dec**2 + self.p2_dec**2)
   
        input_tmp_1        = torch.cat([input.view(self.args.batch_size, 1, self.args.block_len),
                                    torch.zeros((self.args.batch_size, 1, 2*self.args.block_len)).to(device)+0.5], dim=2)  

        x_fwd_1, h_enc_tmp_1  = self.enc_rnn_fwd(input_tmp_1)
        x_tmp_1         = self.enc_act(self.enc_linear(x_fwd_1))

        enc1 = x_tmp_1
        if use_precomp:
            x_tmp_1   = (x_tmp_1 - mu_1_enc)*1.0/v_1_enc

        else:
            x_tmp_1  = self.power_constraint(x_tmp_1)

        x_tmp_1 = np.sqrt(3)*(self.p1_enc*x_tmp_1)/den_enc

        x_rec_1 = x_tmp_1  + fwd_noise[:,:,:,0].view(self.args.batch_size, 1, self.args.block_len)
        x_dec_1, h_dec_tmp_1  = self.dec_rnn(x_rec_1)
        x_dec_1 = self.dec_act(self.dec_output(x_dec_1))
        dec1 = x_dec_1
        if use_precomp:
            x_dec_1   = (x_dec_1 - mu_1_dec)*1.0/v_1_dec

        else:
            x_dec_1  = self.power_constraint(x_dec_1)
        x_dec_1 = np.sqrt(2)*(self.p1_dec*x_dec_1)/den_dec

        noisy_x_dec_1 = x_dec_1 + fb_noise[:,:,:,0].view(self.args.batch_size, 1, self.args.block_len)
        input_tmp_2        = torch.cat([input.view(self.args.batch_size, 1, self.args.block_len),
                                    noisy_x_dec_1.view(self.args.batch_size, 1, self.args.block_len),
                                    torch.zeros((self.args.batch_size, 1, self.args.block_len)).to(device)+0.5], dim=2)  

        x_fwd_2, h_enc_tmp_2  = self.enc_rnn_fwd(input_tmp_2, h_enc_tmp_1)
        x_tmp_2         = self.enc_act(self.enc_linear(x_fwd_2)) 


        enc2 = x_tmp_2

        if use_precomp:
            x_tmp_2   = (x_tmp_2 - mu_2_enc)*1.0/v_2_enc

        else:
            x_tmp_2  = self.power_constraint(x_tmp_2)
        x_tmp_2 = np.sqrt(3)*(self.p2_enc*x_tmp_2)/den_enc


        x_rec_2 = x_tmp_2  + fwd_noise[:,:,:,1].view(self.args.batch_size, 1, self.args.block_len)
        x_dec_2, h_dec_tmp_2  = self.dec_rnn(x_rec_2,h_dec_tmp_1)

        ##before power norm
        x_dec_2_before = self.dec_output(x_dec_2)

        x_dec_2 = self.dec_act(self.dec_output(x_dec_2))
        dec2 = x_dec_2
        if use_precomp:
            x_dec_2   = (x_dec_2 - mu_2_dec)*1.0/v_2_dec

        else:
            x_dec_2  = self.power_constraint(x_dec_2)

        x_dec_2 = np.sqrt(2)*(self.p2_dec*x_dec_2)/den_dec

        noisy_x_dec_2 = x_dec_2 + fb_noise[:,:,:,1].view(self.args.batch_size, 1, self.args.block_len)
        input_tmp_3        = torch.cat([input.view(self.args.batch_size, 1, self.args.block_len),
                                    noisy_x_dec_1.view(self.args.batch_size, 1, self.args.block_len),
                                    noisy_x_dec_2.view(self.args.batch_size, 1, self.args.block_len)], dim=2) 


        x_fwd_3, h_enc_tmp_3  = self.enc_rnn_fwd(input_tmp_3, h_enc_tmp_2)
        x_tmp_3         = self.enc_act(self.enc_linear(x_fwd_3)) 


        enc3 = x_tmp_3

        if use_precomp:
            x_tmp_3   = (x_tmp_3 - mu_3_enc)*1.0/v_3_enc

        else:
            x_tmp_3  = self.power_constraint(x_tmp_3)
        x_tmp_3 = np.sqrt(3)*(self.p3_enc*x_tmp_3)/den_enc

        x_rec_3 = x_tmp_3  + fwd_noise[:,:,:,2].view(self.args.batch_size, 1, self.args.block_len)
        x_dec_3, h_dec_tmp_3  = self.dec_rnn(x_rec_3,h_dec_tmp_2)
        x_dec_3 = self.dec_output(x_dec_3)
        #x_dec_3 = self.power_constraint(x_dec_3)

       


        #final_x=x_dec
        final_x=F.sigmoid(x_dec_3)

        if eval==True:
            return input_tmp_1,x_tmp_1,fwd_noise[:,:,:,0],x_rec_1,x_dec_1,fb_noise[:,:,:,0]   ,input_tmp_2,x_tmp_2,fwd_noise[:,:,:,1],x_rec_2,x_dec_2,x_dec_2_before,fb_noise[:,:,:,1]   ,input_tmp_3,x_tmp_3,fwd_noise[:,:,:,2],x_rec_3,final_x,x_dec_3

 
        if precomp:
            return enc1,enc2,enc3,dec1,dec2

        return final_x


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
        fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
        fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)

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

    num_train_block =  args.num_block

    
    test_ber=.0

    with torch.no_grad():
        mu_1_enc=0.0
        v_1_enc = 0.0
        mu_2_enc=0.0
        v_2_enc = 0.0
        mu_3_enc=0.0
        v_3_enc = 0.0

        mu_1_dec=0.0
        v_1_dec = 0.0
        mu_2_dec=0.0
        v_2_dec = 0.0 

        if args.precompute_stats:
            ##Step 1: save mean and var
            num_test_batch = 5000
            for batch_idx in range(num_test_batch):

                X_test     = torch.randint(0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float)
                fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
                fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)

                # use GPU
                X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

                enc_1,enc_2,enc_3,dec_1,dec_2= model(X_test, fwd_noise, fb_noise, False, True, False)

                mu_1_enc += torch.mean(enc_1)
                v_1_enc += torch.std(enc_1)
                mu_2_enc += torch.mean(enc_2)
                v_2_enc += torch.std(enc_2)
                mu_3_enc += torch.mean(enc_3)
                v_3_enc += torch.std(enc_3)

                mu_1_dec += torch.mean(dec_1)
                v_1_dec += torch.std(dec_1)
                mu_2_dec += torch.mean(dec_2)
                v_2_dec += torch.std(dec_2)

            mu_1_enc /= num_test_batch
            v_1_enc /= num_test_batch
            mu_2_enc /= num_test_batch
            v_2_enc /= num_test_batch
            mu_3_enc /= num_test_batch
            v_3_enc /= num_test_batch

            mu_1_dec /= num_test_batch
            v_1_dec /= num_test_batch
            mu_2_dec /= num_test_batch
            v_2_dec /= num_test_batch

            # print(mu_1_enc)
            # print(v_1_enc)
            # print(mu_2_enc)
            # print(v_2_enc)
            # print(mu_3_enc)
            # print(v_3_enc)

            # print(mu_1_dec)
            # print(v_1_dec)
            # print(mu_2_dec)
            # print(v_2_dec)

            
        

        ##Step 2: compute ber
        num_test_batch = 5000
        for batch_idx in range(num_test_batch):

            X_test     = torch.randint(0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float)
            fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
            fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)

            # use GPU
            X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

            X_hat_test = model(X_test, fwd_noise, fb_noise, False, False, args.precompute_stats, mu_1_enc,v_1_enc,mu_2_enc,v_2_enc,mu_3_enc,v_3_enc,mu_1_dec,v_1_dec,mu_2_dec,v_2_dec)
            
            test_ber  += errors_ber(X_hat_test.cpu(),X_test.cpu())
            

    test_ber  /= 1.0*num_test_batch
    print('Test SNR',frwd_snr ,'with ber ', float(test_ber))


train_model=False
test_ber=False
eval_scheme=True   

##Training
if train_model:

    for epoch in range(1, args.num_epoch + 1):
        train(epoch)

        if epoch%10 == 0:
            torch.save(model.state_dict(), "./power_norm_models_bl1_snr2/trainable_weights_16.pt")
            print("Model is saved")
            test_2(model)

##Testing
elif test_ber:

    pretrained_model = torch.load("./models/power_norm_models_bl1_snr0/trainable_weights_16.pt")
    model.load_state_dict(pretrained_model)
    model.args = args
    test_2(model)

##Scheme analysis
elif eval_scheme:    
    pretrained_model = torch.load("./models/power_norm_models_bl1_snr0/trainable_weights_16.pt")
    model.load_state_dict(pretrained_model)
    model.args = args

    model.eval()
    frwd_snr = args.frwd_snr
    frwd_sigma = 10**(-frwd_snr*1.0/20)

    bckwd_snr = args.bckwd_snr
    bckwd_sigma = 10**(-bckwd_snr*1.0/20)

    num_train_block =  args.num_block

    with torch.no_grad():
        mu_1_enc=0.0
        v_1_enc = 0.0
        mu_2_enc=0.0
        v_2_enc = 0.0
        mu_3_enc=0.0
        v_3_enc = 0.0

        mu_1_dec=0.0
        v_1_dec = 0.0
        mu_2_dec=0.0
        v_2_dec = 0.0 

        if args.precompute_stats:
            ##Step 1: save mean and var
            num_test_batch = 5000
            for batch_idx in range(num_test_batch):

                X_test     = torch.randint(0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float)
                fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
                fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)

                # use GPU
                X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

                enc_1,enc_2,enc_3,dec_1,dec_2= model(X_test, fwd_noise, fb_noise, False, True, False)

                mu_1_enc += torch.mean(enc_1)
                v_1_enc += torch.std(enc_1)
                mu_2_enc += torch.mean(enc_2)
                v_2_enc += torch.std(enc_2)
                mu_3_enc += torch.mean(enc_3)
                v_3_enc += torch.std(enc_3)

                mu_1_dec += torch.mean(dec_1)
                v_1_dec += torch.std(dec_1)
                mu_2_dec += torch.mean(dec_2)
                v_2_dec += torch.std(dec_2)

            mu_1_enc /= num_test_batch
            v_1_enc /= num_test_batch
            mu_2_enc /= num_test_batch
            v_2_enc /= num_test_batch
            mu_3_enc /= num_test_batch
            v_3_enc /= num_test_batch

            mu_1_dec /= num_test_batch
            v_1_dec /= num_test_batch
            mu_2_dec /= num_test_batch
            v_2_dec /= num_test_batch

        
        

        ##Step 2: generate examples
        num_test_batch = 1
        for batch_idx in range(num_test_batch):

            X_zeros=torch.zeros((args.batch_size//2, 1, args.block_len), dtype=torch.float)
            X_ones=torch.ones((args.batch_size//2, 1, args.block_len), dtype=torch.float)
            X_test=torch.cat([X_zeros,X_ones],dim=0)
            fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
            fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)

            # use GPU
            X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)
        
            msg1,code1,fwnoise1,x_rec_1,fb1,fbnoise1,msg2,code2,fwnoise2,x_rec_2,fb2,fb2_before,fbnoise2,msg3,code3,fwnoise3,x_rec_3,decoded_post_sigmoid,decoded_pre_sigmoid = model(X_test, fwd_noise, fb_noise, True, False, args.precompute_stats, mu_1_enc,v_1_enc,mu_2_enc,v_2_enc,mu_3_enc,v_3_enc,mu_1_dec,v_1_dec,mu_2_dec,v_2_dec)


        ##Round 1
        msg1=np.reshape(msg1.cpu().numpy(),(args.batch_size,3))
        code1=np.reshape(code1.cpu().numpy(),(args.batch_size,1))
        fwnoise1=np.reshape(fwnoise1.cpu().numpy(),(args.batch_size,1))
        fb1=np.reshape(fb1.cpu().numpy(),(args.batch_size,1))
        x_rec_1=np.reshape(x_rec_1.cpu().numpy(),(args.batch_size,1))
        fbnoise1=np.reshape(fbnoise1.cpu().numpy(),(args.batch_size,1))


        ##Round 2
        msg2=np.reshape(msg2.cpu().numpy(),(args.batch_size,3))
        code2=np.reshape(code2.cpu().numpy(),(args.batch_size,1))
        fwnoise2=np.reshape(fwnoise2.cpu().numpy(),(args.batch_size,1))
        fb2=np.reshape(fb2.cpu().numpy(),(args.batch_size,1))
        fb2_before=np.reshape(fb2_before.cpu().numpy(),(args.batch_size,1))
        x_rec_2=np.reshape(x_rec_2.cpu().numpy(),(args.batch_size,1))
        fbnoise2=fbnoise2.cpu().numpy()


        ##Round 3
        msg3=np.reshape(msg3.cpu().numpy(),(args.batch_size,3))
        code3=np.reshape(code3.cpu().numpy(),(args.batch_size,1))
        fwnoise3=fwnoise3.cpu().numpy()
        x_rec_3=np.reshape(x_rec_3.cpu().numpy(),(args.batch_size,1))
        mdecoded_post_sigmoidsg1=np.reshape(decoded_post_sigmoid.cpu().numpy(),(args.batch_size,1))
        decoded_pre_sigmoid=np.reshape(decoded_pre_sigmoid.cpu().numpy(),(args.batch_size,1))



    ##Encoder

    fig = plt.figure()

    plt.figure(1)
    plt.plot(msg1[:args.batch_size//2,0],code1[:args.batch_size//2],'bo',label='initial msg=0')
    plt.plot(msg1[args.batch_size//2:,0],code1[args.batch_size//2:],'r+',label='initial msg=1')
    plt.xlabel('$b$',fontsize=12)
    plt.ylabel('$c^1$',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis1.png')

    plt.figure(2)
    plt.plot(msg2[:args.batch_size//2,1],code2[:args.batch_size//2],'bo',label='initial msg=0')
    plt.plot(msg2[args.batch_size//2:,1],code2[args.batch_size//2:],'r+',label='initial msg=1')
    plt.xlabel('$\hat{b}^1 + w^1$',fontsize=12)
    plt.ylabel('$c^2$',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis2.png')

    plt.figure(3)
    plt.plot(msg3[:args.batch_size//2,2],code3[:args.batch_size//2],'bo',label='initial msg=0')
    plt.plot(msg3[args.batch_size//2:,2],code3[args.batch_size//2:],'r+',label='initial msg=1')
    plt.xlabel('$\hat{b}^2 + w^2$',fontsize=12)
    plt.ylabel('$c^3$',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis3.png')


    ##Decoder
    plt.figure(4)
    plt.plot(x_rec_1[:args.batch_size//2],fb1[:args.batch_size//2],'bo',label='initial msg=0')
    plt.plot(x_rec_1[args.batch_size//2:],fb1[args.batch_size//2:],'r+',label='initial msg=1')
    plt.xlabel('$y^1$',fontsize=12)
    plt.ylabel('$\hat{b}^1$',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis4.png')

    plt.figure(5)
    plt.plot(x_rec_2[:args.batch_size//2],fb2[:args.batch_size//2],'bo',label='initial msg=0')
    plt.plot(x_rec_2[args.batch_size//2:],fb2[args.batch_size//2:],'r+',label='initial msg=1')
    plt.xlabel('Received Value Round 2',fontsize=12)
    plt.ylabel('Decoder 2 Output',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis5.png')

    plt.figure(6)
    plt.plot(x_rec_3[:args.batch_size//2],mdecoded_post_sigmoidsg1[:args.batch_size//2],'bo',label='initial msg=0')
    plt.plot(x_rec_3[args.batch_size//2:],mdecoded_post_sigmoidsg1[args.batch_size//2:],'r+',label='initial msg=1')
    plt.xlabel('$y^3$',fontsize=12)
    plt.ylabel('$\hat{b}^3$',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis6.png')


    ##Color code last step
    x_rec_3_temp1 = []
    mdecoded_post_sigmoidsg1_temp1 = []

    x_rec_3_temp2 = []
    mdecoded_post_sigmoidsg1_temp2 = []

    for i in range(args.batch_size):
        if fb2[i,:] <= 0:
            x_rec_3_temp1.append(x_rec_3[i])
            mdecoded_post_sigmoidsg1_temp1.append(mdecoded_post_sigmoidsg1[i])
        else:
            x_rec_3_temp2.append(x_rec_3[i])
            mdecoded_post_sigmoidsg1_temp2.append(mdecoded_post_sigmoidsg1[i])

    plt.figure(7)
    plt.plot(x_rec_3_temp1,mdecoded_post_sigmoidsg1_temp1,'bo',label='$\hat{b}^2$<=0')
    plt.plot(x_rec_3_temp2,mdecoded_post_sigmoidsg1_temp2,'r+',label='$\hat{b}^2$>0')
    plt.xlabel('$y^3$',fontsize=12)
    plt.ylabel('$\hat{b}^3$',fontsize=12)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('analysis7.png')




    

