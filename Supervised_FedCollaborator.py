"""
Supervised_FedCollaborator.py

Class for running local training (FedDis); delivered to client

"""
import torch
import numpy as np
import wandb
import logging
from time import time

from monai.losses import DiceCELoss
from monai.optimizers import Adam
from monai.optimizers.lr_scheduler import ExponentialLRDecay
from monai.transforms import ToRGB
from monai.ILIA.core.FedCollaborator import FedCollaborator


class Supervised_FedDisCollaborator(FedCollaborator):
    def __init__(self, client_idx, client_name, training_params, model, device, log_wandb=False):
        """
        Init function for Client
        :param client_idx: int
            Id of the client
        :param client_name: str
            Name of the client
        :param training_params: list
            parameters for local training routine
        :param device: torch.device
            GPU  |  CPU
        :param model: torch.nn.module
            Neural  network
        """
        super(Supervised_FedDisCollaborator, self).\
            __init__(client_idx=client_idx, client_name=client_name, training_params=training_params, model=model,
                     device=device, log_wandb=log_wandb)

        self.eval_step = 0
        self.min_val_loss = np.inf
        self.train_step = 0

        self.criterion = DiceCELoss(ce_weight=torch.cuda.FloatTensor([0.8]))

        self.img2RGB = ToRGB(255, 255, 255)

        self.optimizer = Adam(self.model.parameters(), lr=training_params['optimizer_params']['learning_rate'])
        self.decayRate = training_params['optimizer_params']['decay_rate']
        self.my_lr_scheduler = ExponentialLRDecay(optimizer=self.optimizer, gamma=self.decayRate)

    def load_data(self, dataset_name):
        """
        :param dataset_name: str
            name of the dataset(s) to load
        :return:
            local_training_data: monai.DataLoader
            local_test_data: monai.DataLoader
            num_train_samples: int
                number of training samples
        """
        if not isinstance(dataset_name, list):
            dataset_name = [dataset_name]
        local_training_data = self.data_loader.load_data(dataset_current_list=dataset_name)
        return local_training_data, local_training_data

    def get_nr_train_samples(self):
        return len(self.local_training_data) * self.local_training_data.batch_size

    def train(self, w_global, opt_state, round_idx=0):
        """
        Train local client
        :param w_global: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param round_idx: int
            the round number
        :return:
            self.model.state_dict():
            self.optimizer.state_dict():
            epoch_loss: float
                the loss of the epoch
        """
        # return self.client_name, self.model.state_dict(), self.optimizer.state_dict(), epoch_loss
        for param in self.model.state_dict():
            check_equal = torch.equal(self.model.state_dict()[param], w_global[param])
            if not check_equal:
                logging.info("************************** model params equal?   ({}): {}".format(param, 'different'))

        self.model.load_state_dict(w_global)  # load weights
        self.model.train()
        self.model.to(self.device)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        # self.optimizer.state = collections.defaultdict(dict) #  reset optimizer

        epoch_losses = []
        for epoch in range(self.training_params['nr_epochs']):
            start_time = time()
            batch_loss = 1.0
            count_images = 0
            ct_train = np.floor(int(0.7 * len(self.local_training_data)))
            print('Number of training batches: {}'.format(ct_train))
            ct = 0
            for data in self.local_training_data:
                # Input
                ct += 1
                if ct > ct_train:
                    break
                input_all_slices = data[0]
                gt_all_slices = data[1]

                images = input_all_slices
                gt = gt_all_slices
                b, sl, w, h = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
                images = images.view(b*sl, 1,  w, h)
                gt = gt.view(b*sl, 1, w, h)

                count_images += images.shape[0]
                images, gt = images.to(self.device), gt.to(self.device)

                # Forward Pass
                self.optimizer.zero_grad()
                segmented_images = self.model(images)['x_rec']

                # Reconstruction Loss
                loss = self.criterion(segmented_images, gt)

                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item() * images.size(0)

                if count_images >= self.training_params['max_iterations']:
                    logging.info('Break due to maximal number of local iterations...')
                    break
            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss

            epoch_losses.append(epoch_loss)
            end_time = time()
            mid_slice = int(sl / 2 / b)
            grid_image = np.hstack([self.img2RGB(images.detach().cpu()[mid_slice].numpy())[0, :, :128],
                                    self.img2RGB(gt.detach().cpu()[mid_slice].numpy())[0, :, :128],
                                    self.img2RGB(segmented_images.detach().cpu()[mid_slice].numpy())[0, :, :128]])

            wandb.log({"Train\Example_" + self.client_name: [
                wandb.Image(grid_image, caption="Reconstruction_" + str(self.train_step))]})
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))

            wandb.log({"Train/Loss_" + self.client_name: epoch_loss, '_step_' + self.client_name: self.train_step})

            torch.save(self.model.state_dict(), self.client_path + '/latest_model.pt')
            self.train_step += 1

            if epoch % 3 == 0:
                self.test(self.model.state_dict())

        return self.client_name, self.model.state_dict(), self.optimizer.state_dict(), sum(epoch_losses) / len(
            epoch_losses)

    def test(self, model_weights):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_l1': 0,
                'test_total': 0
            }
        """
        # return metrics
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            'test_loss': 0,
        }
        test_total = 0
        with torch.no_grad():
            ct_test = np.floor(int(0.7 * len(self.local_training_data)))
            ct = 0
            for data in self.local_test_data:
                ct += 1
                if ct <= ct_test:
                    continue
                x_all = data[0]
                gt_all = data[1]
                x = x_all
                gt = gt_all
                b, sl, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
                x = x.view(b * sl, 1, w, h)
                gt = gt.view(b * sl, 1, w, h)
                test_total += x.shape[0]
                x, gt = x.to(self.device), gt.to(self.device)

                # Forward pass
                y, _ = self.test_model(x)
                loss = self.criterion(y, gt)

                metrics['test_loss'] += loss.item() * x.size(0)
        mid_slice = int(x.shape[0] / 2 / b)

        grid_image = np.hstack([self.img2RGB(x.detach().cpu()[mid_slice].numpy())[0, :, :128],
                                self.img2RGB(gt.detach().cpu()[mid_slice].numpy())[0, :, :128],
                                self.img2RGB(y.detach().cpu()[mid_slice].numpy())[0, :, :128]])
        wandb.log({"Test\Example_" + self.client_name: [wandb.Image(
            grid_image, caption="Reconstruction_" + str(self.train_step))]})

        for metric_key in metrics.keys():
            metric_name = 'Test/' + str(metric_key) + '_' + self.client_name
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_' + self.client_name: self.train_step})

        if loss < self.min_val_loss:
            self.min_val_loss = loss
            torch.save(model_weights, self.client_path + '/best_model.pt')
        self.my_lr_scheduler.step()

        return metrics, test_total
