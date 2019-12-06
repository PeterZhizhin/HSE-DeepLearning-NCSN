import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

import toy_experiment_dataset
import model_mlp
from sliced_sm import sliced_score_estimation_vr

def langevin(probs, means, num, input, lr=0.01, step=1000):
	ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, 1)
	for i in range(step):
		#print(input.dtype)
		input +=  lr * ds.compute_p_gradient(input).float().detach()/2
		input += torch.randn_like(input) * np.sqrt(lr)
		#print(input)
	return input

def anneal_langevin(probs, means, num, input, sigmas, lr=0.01, step=1000):
	for s in sigmas:
		ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, 1, sigma=s)
		for i in range(step):
			#print(s, sigmas[-1])
			lr_new = lr * np.power(s / sigmas[-1], 2)
			input += lr_new * ds.compute_p_gradient(input).float().detach()/2
			input += torch.randn_like(input) * np.sqrt(lr)
		#print(input)
	return input

def toy_generate():
	probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
	means = np.array([[-5, -5], [5, 5]], dtype=np.float)
	num = 1280

	ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)

	dataset = ds.tensors[0].numpy()
    # Fig3a samples
	d1 = (dataset[:, 0] < 0).sum()/1280
	d2 = (dataset[:, 0] > 0).sum()/1280
	plt.scatter(dataset[dataset[:, 0] < 0, 0], dataset[dataset[:, 1] < 0, 1], s=1, label='Доля = ' + str(d1))
	plt.scatter(dataset[dataset[:, 0] > 0, 0], dataset[dataset[:, 1] > 0, 1], s=1, label='Доля = ' + str(d2))
	plt.title('Samples')
	plt.legend()
	#plt.savefig("figures/fig3a")
	plt.show()
    

	# Fig3b

	#ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
	start_point = torch.rand(1280, 2) * 16 - 8
	#print(start_point)
	after_lan  = langevin(probs, means, num, start_point, lr = 0.1, step=1000).detach().numpy()
	d1 = (after_lan[:, 0] < 0).sum()/1280
	d2 = (after_lan[:, 0] > 0).sum()/1280
	plt.scatter(after_lan[after_lan[:, 0] < 0, 0], after_lan[after_lan[:, 1] < 0, 1], s=1, label='Доля = ' + str(d1))
	plt.scatter(after_lan[after_lan[:, 0] > 0, 0], after_lan[after_lan[:, 1] > 0, 1], s=1, label='Доля = ' + str(d2))
	plt.title('Langevin')
	plt.legend()
	#plt.savefig("figures/fig3b")
	plt.show()

	# Fig3c

	sigmas = np.geomspace(20, 0.7, 10)#np.geomspace(2, 0.1, 10)#np.exp(np.linspace(np.log(20), 0., 10))#np.geomspace(10, 0.1, 10)
	#print(sigmas)
	#ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
	start_point = torch.rand(1280, 2) * 16 - 8
	after_lan  = anneal_langevin(probs, means, num, start_point, sigmas, lr = 0.1, step=100).detach().numpy()
	d1 = (after_lan[:, 0] < 0).sum()/1280
	d2 = (after_lan[:, 0] > 0).sum()/1280
	plt.scatter(after_lan[after_lan[:, 0] < 0, 0], after_lan[after_lan[:, 1] < 0, 1], s=1, label='Доля = ' + str(d1))
	plt.scatter(after_lan[after_lan[:, 0] > 0, 0], after_lan[after_lan[:, 1] > 0, 1], s=1, label='Доля = ' + str(d2))
	plt.title('Annealed Langevin')
	plt.legend()
	#plt.savefig("figures/fig3c")
	plt.show()



def train():
	model = model_mlp.ModelMLP(128)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	probs = np.array([1.0 / 5, 4.0 / 5], dtype=np.float)
	means = np.array([[-5, -5], [5, 5]], dtype=np.float)
	num = 1280000
	toy_ds = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
	toy_dl = torch.utils.data.DataLoader(toy_ds, batch_size=128)
	for points in toy_dl:
		#input = toy_experiment_dataset.ToyExperimentDataset(probs, means, num)
		#print(points)
		loss, _, _ = sliced_score_estimation_vr(model, points[0].float())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return model

def main():
	#model = train()
	toy_generate()




















