from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.5
    std = 255
    inp = std * inp + mean
    inp = np.clip(inp.astype(int), 0, 255)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.plot([1, 2], [1, 2])
    plt.show() 
    #plt.pause(0.1)


def data_lavgevin(input, model, lr=0.01, step=1000, n_sigma = 9):
	#res_im = []
	num_sigmas = torch.ones(input.shape[0]).long() * n_sigma
	for i in range(step):
		#res_im.append(torch.clamp(input, 0., 1.))
		input +=  lr * model(input, num_sigmas)/2
		input += torch.randn_like(input) * np.sqrt(lr)
	return input

def data_anneal_lavgevin(input, model, sigmas, lr=0.01, step=1000):
	#res_im = []
	for k, s in enumerate(sigmas):
		num_sigmas = torch.ones(input.shape[0]).long() * k
		for i in range(step):
			lr_new = lr * np.power(s / sigmas[-1], 2)
			input += lr_new * model(input, num_sigmas)/2
			input += torch.randn_like(input) * np.sqrt(lr_new)
		#print(input)
	return input


def generate_MNIST_vanilla(model, batch): #
	model.eval()
	start_point = torch.rand(batch * batch, 1, 28, 28)
	after_lan = data_lavgevin(start_point, model, lr=2 * 1e-5, step=1000, n_sigma = 9)
	after_lan = start_point
	grid = make_grid(after_lan, nrow=batch)
	imshow(grid)



def generate_MNIST_anneal(model, sigmas, batch):
	model.eval()
	start_point = torch.rand(batch * batch, 1, 28, 28)
	after_lan = data_anneal_lavgevin(start_point, model, sigmas, lr=5 * 1e-5, step=100)
	after_lan = start_point
	grid = make_grid(after_lan, nrow=batch)
	imshow(grid)










