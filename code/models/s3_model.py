import numpy as np
import torch
import torch.nn as nn
import timm
import ttach as tta


def test_(net, test_dataset,tot_num, tta_aug, device): 
    list_tot = np.zeros((tot_num))
    list_lab1 = np.zeros((tot_num))
    list_real = np.zeros((tot_num))

    transforms = tta.Compose(tta_aug)
    for transformer in transforms:
        with torch.no_grad():
            net.eval()
            ct = 0
            for input, label, feature, idx2 in test_dataset:
                ct = ct+len(label[label==1])
                input = transformer.augment_image(input)
                input= input.to(device)
                feature= feature.to(device)
                label= label.to(device) 
            
                logps = net(input, feature, device)
                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)
                #equals = top_class == label.view(*top_class.shape)
                list_tot[idx2] += 1 
                if top_class[0][0] == 1: 
                    list_lab1[idx2] +=1
                if label[0] == 1: 
                    list_real[idx2] = 1
    test_out = []
    for ix in range(tot_num):
        if list_tot[ix] - list_lab1[ix] > list_lab1[ix]: 
            an = 0
        else:
            an = 1
        test_out.append(an)    
    return test_out, list_tot, list_lab1, list_real


# model
class MyModel(nn.Module):
    def __init__(self, data_name, model, selected_features):
        super(MyModel, self).__init__()
        
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
            def forward(self, x):
                return x
            
        self.res_model_g = timm.create_model('resnet18',in_chans=1, num_classes = 2)
        self.res_model_g_fc = timm.create_model('resnet18',in_chans=1, num_classes = 2)
        self.res_model_g_fc.fc = Identity()
        self.res_model_c = timm.create_model('resnet18',in_chans=3, num_classes = 2)
        self.res_model_c_fc = timm.create_model('resnet18',in_chans=3, num_classes = 2)
        self.res_model_c_fc.fc = Identity()
        
        self.class512t16 = nn.Linear(512,16)
        self.class48t2 = nn.Linear(48,2)
        self.class32t2 = nn.Linear(32,2)

        self.classfeaturest16 = nn.Linear(len(selected_features),16)
        self.data_name = data_name
        self.model = model

        self.selected_features = selected_features

    def forward(self, xinput, xfeature = [], device='cpu'):
        if self.data_name != 'PA_VHE': 
            if len(xfeature[0,:]) == 0:
                if self.data_name == 'PA':
                    x = self.res_model_g(xinput)
                else:
                    x = self.res_model_c(xinput)
            else:
                if self.data_name == 'PA':
                    x = self.res_model_g_fc(xinput)
                else:
                    x = self.res_model_c_fc(xinput)
                x = self.class512t16(x)
                
                new_features = torch.Tensor(np.zeros((xfeature.shape[0], len(self.selected_features)))).to(device)
                for ix, y in enumerate(self.selected_features):
                    new_features[:,ix] = xfeature[:, int(y)]

                xfeature = self.classfeaturest16(new_features)
                x = torch.cat([x, xfeature], dim=1)
                x = self.class32t2(x)
        else: 
            if self.model == 'basic':
                x1 = self.res_model_g_fc(xinput[:,0:1,:,:])
                x1 = self.class512t16(x1)
                x2 = self.res_model_c_fc(xinput[:,1:4,:,:])
                x2 = self.class512t16(x2)
                if len(xfeature[0,:]) == 0:
                    x = torch.cat([x1,x2], dim=1)
                    x = self.class32t2(x)
                else:
                    new_features = torch.Tensor(np.zeros((xfeature.shape[0], len(self.selected_features)))).to(device)
                    for ix, y in enumerate(self.selected_features):
                        new_features[:,ix] = xfeature[:, int(y)]
                    xfeature = self.classfeaturest16(new_features)
                    x = torch.cat([x1,x2,xfeature], dim=1)
                    x = self.class48t2(x)
        return x