from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import params

#def dsm_loss(model, inputs, labels, sigmas, power=2):
#	re_sigma = sigmas[labels].view(inputs.shape[0], 1, 1, 1)
#	sam_noise = inputs + torch.rand_like(inputs) * re_sigma
#	target = -(sam_noise - inputs)/torch.pow(re_sigma, 2)
#	res = model(sam_noise, labels)
#	loss = 0.5 * (torch.pow(target - res, 2)).sum(dim = [1, 2, 3]) * torch.pow(re_sigma, power)
#	loss = torch.mean(loss)
#	return loss

def train_MNIST(batch_size, n_epoch):
    
    if params.dataset == 'MNIST':
	    data_transforms = {
	    'train': transforms.Compose([
	        transforms.Resize(256),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor()
	    ]),
	    'test': transforms.Compose([
	        transforms.Resize(256),
	        transforms.ToTensor()
	    ])
	    }

	    train_data = MNIST('data/MNIST/train', train=True, download=True, transform=data_transforms['train'])
	    test_data = MNIST('data/MNIST/test', train=False, download=True, transform=data_transforms['train'])

	train_loader = Dataloader(train_data, batch_size, shuffle=True, num_workers=4)
	test_loader = Dataloader(test_data, batch_size, shuffle=True, num_workers=4)
	dataloaders['train'] = train_loader
	dataloaders['test'] = test_loader
	model = #MODEL!
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0, betas=(0.9, 0.999))

	sigmas = torch.exp(torch.linspace(torch.log(1), torch.log(0.01), 10))

    step = 0
    n_iter = # n_iter
	for epoch in range(n_epoch):
		for phase in ['train', 'test']:
			if phase == 'train':
                scheduler.step()
                model.train(True) 
            else:
                model.train(False)  

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
            	step += 0.5

                inputs, labels = data

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs = inputs/256. * 255 + torch.rand_like(inputs) / 256.

                loss_lab = torch.randint(0, len(sigmas), (inputs.shape[0], ))
                loss = # dsm_annealed (model, inputs, loss_lab, sigmas, 2)

                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if step > n_iter:
                	#??










