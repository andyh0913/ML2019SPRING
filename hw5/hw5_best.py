import numpy as np
import os
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import resnet50


epsilon = 0.06

# model = resnet50(pretrained=True)
model = resnet50(pretrained=True)

model.eval()
criterion = nn.CrossEntropyLoss()

def load_data(folder_path="/content/drive/My Drive/ML2019Spring/hw5/data/images", labels_path="/content/drive/My Drive/ML2019Spring/hw5/data/labels.csv"):
    if os.path.isfile("./data/img_list.npy"):
        print ("Read data from img_list.npy...")
        img_list = np.load("./data/img_list.npy")
    else:
        print ("Preprocessing data...")
        normalize = transform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        preprocess = transform.Compose([transform.ToTensor(),normalize])
        
        img_list = []
        for i in range(200):
            img_path = os.path.join(folder_path, str(i).zfill(3) + ".png")
            img = Image.open(img_path)
            img = preprocess(img).numpy()
            print (img.shape)
            img_list.append(img)
        img_list = np.array(img_list)
        np.save("./data/img_list.npy", img_list)
    labels = np.genfromtxt(labels_path, delimiter=',')[1:,3:4]
    return img_list, labels.astype(int)

if __name__ == '__main__':
	input_path = sys.argv[1]
    output_path = sys.argv[2]
    mean=[0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x,labels = load_data(input_path)

    inverse_normalize = transform.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],std=[1/0.229, 1/0.224, 1/0.255])
    postprocess = transform.Compose([inverse_normalize])

    x_t = torch.from_numpy(x)
    labels_t = torch.from_numpy(labels)

    # fgsm
    for i in range(labels.shape[0]):
        print ("Processing "+str(i)+"th image...")
        image = x_t[i].unsqueeze(0)
        image.requires_grad = True
        zero_gradients(image)

        output = model(image)
        loss = criterion(output, labels_t[i])
        loss.backward()

        image = image + epsilon * image.grad.sign_()
        
        image = image[0]
        image = image.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1))

#         image = postprocess(image[0])
        image = torch.clamp(image,0,1).detach().numpy()
        image = image.transpose(1,2,0)       

        img = Image.fromarray((image*255).astype(np.uint8), mode="RGB")
        img_path = os.path.join(output_path, str(i).zfill(3) + ".png")
        img.save(img_path)



	# print ("success rate:", success_rate)	
