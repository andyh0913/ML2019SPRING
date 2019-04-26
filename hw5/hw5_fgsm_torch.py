import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import resnet50

epsilon = 0.3

model = resnet50(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()

def load_data(folder_path="data/images", labels_path="data/labels.csv"):
	if os.path.isfile("data/img_list.npy"):
		img_list = np.load("data/img_list.npy")
	else:
		img_list = []
		for i in range(200):
			img_path = os.path.join(folder_path, str(i).zfill(3) + ".png")
			img = image.load_img(img_path, target_size=(224,224))
			img_list.append(image.img_to_array(img))
		img_list = np.array(img_list)
		np.save("data/img_list.npy", img_list)
	labels = np.genfromtxt(labels_path, delimiter=',')[1:,3:4]
	return img_list, labels

if __name__ == '__main__':
	output_path = "output"


	x,labels = load_data()

	normalize = transform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	scale = transform.Normalize(mean=[0,0,0],std=[255.,255.,255.])
	preprocess = transform.Compose([transform.ToTensor(),scale,normalize])
	inverse_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],std=[1/0.229, 1/0.224, 1/0.255])

	trans = transform.Compose([transform.ToTensor()])
	x_t = preprocess(x)
	labels_t = trans(labels)
	
	# fgsm
	for i in range(labels.shape[0]):
		image = x[i].unsqueeze(0)
		image.requires_grad = True
		zero_gradients(image)

		output = model(image)
		loss = criterion(output, labels)
		loss.backward()

		image = image - epsilon * image.grad.sign_()

		image = inverse_normalize(image).numpy()
		img = Image.fromarray(image)
		img_path = os.path.join(output_path, str(i).zfill(3) + ".png")
		img.save(img_path)



	# print ("success rate:", success_rate)	
