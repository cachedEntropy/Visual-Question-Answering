import json
import pickle
import config
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
from torchvision import datasets
from tqdm import tqdm
features_dict = {}
from PIL import Image
import numpy as np

def data_loader(prefix, batch_size=1):
    data_dir = './'
    path = './Questions.json'
    word2id_dict = pickle.load(open('data/dictionary.pkl'))[0]
    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path


    resnet18 = models.resnet101(pretrained=True)
    modules=list(resnet18.children())[:-1]
    resnet18=nn.Sequential(*modules)
    for p in resnet18.parameters():
        p.requires_grad = False
    resnet18 = resnet18.cuda()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    image_dataset = ImageFolderWithPaths(os.path.join(data_dir), data_transforms['train'])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=8)


    for img, lbl, imgpath in tqdm(dataloader):
        # print(path)
        img = Variable(img).cuda()
        features = resnet18(img)
        features_dict[imgpath[0].split('/')[-1]] = np.asarray(features.cpu().numpy().squeeze())
        
    idx = 0
    max_q_len = 30
    qs = json.load(open(path))['quest_answers']
    
    id = idx
    batch_img, batch_wl, batch_index, batch_idx, batch_len = [],[],[],[],[]
    for i,q in enumerate(qs):  
        if ((i+1)%batch_size!=0):
            point_question = q['Question']
            point_index = q['Index']
            point_img = q['Image']

            img_features = features_dict[point_img + '.png']
            point_question = point_question.replace(',', '').replace('?', '').replace('\'s', ' \'s')
            words = point_question.lower().split()
            wl = [word2id_dict[word] for word in words]
            if len(wl) > max_q_len:
                wl = wl[:30]
            else:
                while(len(wl)<30):
                    wl.append(9)


            batch_img.append(img_features)
            batch_wl.append(wl)
            batch_index.append(point_index)
            batch_idx.append(idx)
            batch_len.append(len(wl))    
        else:
            point_question = q['Question']
            point_index = q['Index']
            point_img = q['Image']

            img_features = features_dict[point_img + '.png']
            point_question = point_question.replace(',', '').replace('?', '').replace('\'s', ' \'s')
            words = point_question.lower().split()
            wl = [word2id_dict[word] for word in words]
            if len(wl) > max_q_len:
                wl = wl[:30]
            else:
                while(len(wl)<30):
                    wl.append(9)


            batch_img.append(img_features)
            batch_wl.append(wl)
            batch_index.append(point_index)
            batch_idx.append(idx)
            batch_len.append(len(wl))    
            yield batch_img, batch_wl, batch_index, batch_idx, batch_len
            batch_img, batch_wl, batch_index, batch_idx, batch_len = [],[],[],[],[]
            id = idx
            idx+=1  