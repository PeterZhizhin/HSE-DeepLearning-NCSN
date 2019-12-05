import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

import toy_experiment_dataset
import model_mlp
from sliced_sm import sliced_score_estimation_vr

def langevin(model, input, lr=0.01, step=1000):
	for i in range(step):
		input += lr * model(input).detach()/2
		input += torch.randn_like(input) * torch.sqrt(lr)
	return input

def anneal_langevin(model, input, sigmas, lr=0.01, step=1000):
	for s in sigmas:
		for i in range(step):
			lr_new = lr * torch.pow(s/ s[-1], 2)
			input += lr * model(input).detach()/2
			input += torch.randn_like(input) * torch.sqrt(lr)
	return input

def toy_generate(model):
	probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
	means = np.array([[-5, -5], [5, 5]], dtype=np.float)
	num = 1280

	ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)

	dataset = ds.tensors[0].numpy()
    # Fig3a samples

	plt.scatter(dataset[:, 0], dataset[:, 1])
	plt.title('Samples')
	plt.legend()
	#plt.savefig("figures/fig3a")
	plt.show()
    

	# Fig3b

	start_point = torch.rand(1280, 2) * 16 - 8
	after_lan  = langevin(model, start_point, lr = 0.1, step=1000).detach().numpy()
	plt.scatter(after_lan[:, 0], after_lan[:, 1])
	plt.title('Langevin')
	plt.legend()
	#plt.savefig("figures/fig3b")
	plt.show()

	start_point = torch.rand(1280, 2) * 16 - 8
	sigmas = np.geomspace(10, 0.1, 10)
	after_lan  = anneal_langevin(model, start_point, sigmas, lr = 0.1, step=100).detach().numpy()
	plt.scatter(after_lan[:, 0], after_lan[:, 1])
	plt.title('Annealed Langevin')
	plt.legend()
	#plt.savefig("figures/fig3c")
	plt.show()



def train():
	model = model_mlp.ModelMLP(128)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
	means = np.array([[-5, -5], [5, 5]], dtype=np.float)
	num = 4
	toy_ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
	toy_dl = torch.utils.data.DataLoader(toy_ds, batch_size=4)
	for points in toy_dl:
		#input = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
		#print(points)
		loss, _, _ = sliced_score_estimation_vr(model, points[0].float())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return model

def main():
	model = train()
	toy_generate(model)




















