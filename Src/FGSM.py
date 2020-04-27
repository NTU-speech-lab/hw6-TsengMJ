import os
import pandas as pd
import sys
from PIL import Image
from PIL import ImageFilter
import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda")

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]

        return img, label
    
    def __len__(self):
        return 200


class Attacker:
    def __init__(self, img_dir, label, model):
        # 讀入預訓練模型 vgg16
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                adv_ex = data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                adv_examples.append( (None, None, None, adv_ex) )
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
          
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1

            # 將攻擊的圖片存入
            adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
            data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            data_raw = data_raw.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )        
        final_acc = (fail / (wrong + success + fail))
        
        print('Wrong: {}'.format(wrong))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc



if __name__ == '__main__':
    # 讀入圖片相對應的 label
    print("Read labels")
    df = pd.read_csv(os.path.join(input_dir, "labels.csv"))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(input_dir, "categories.csv"))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    for i in range(len(label_name)):
        label_name[i] = label_name[i].split()[0].split(',')[0]
        
    # new 一個 Attacker class
    print("Load Model")
    model = models.densenet121(pretrained = True)
    print("Attack")
    attacker = Attacker(os.path.join(input_dir, "images"), df, model=model)
    

    epsilons = [0.095]
    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)



import cv2 

print("Save attacked Images")
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        orig,adv,orig_img, ex = examples[i][j]
        ex = np.transpose(ex, (1, 2, 0))                            
        cv2.imwrite(os.path.join(output_dir, "{:03d}.png".format(j)), ex[:,:, [2,1,0]]*255)
