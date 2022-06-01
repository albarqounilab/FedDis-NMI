"""
FedCollaborator.py

Class for running local training (FedDis); -> delivered to client

"""

import torch
import wandb
import copy
from time import time

from monai.losses import L1Loss, CosineEmbeddingLoss
from monai.optimizers import Adam
from monai.optimizers.lr_scheduler import ExponentialLRDecay
from monai.transforms import RandRectangleMasking, RandAdjustContrast
from monai.ILIA.core.FedCollaborator import FedCollaborator
from monai.ILIA.data.BrainMR.utils import *
from monai.networks.nets import VGGEncoder
from monai.losses import TotalVariationLoss


class FedDisCollaborator(FedCollaborator):
    def __init__(self, client_idx, client_name, training_params, model, device, log_wandb=False):
        """
        Init function for Client
        :param client_idx: int
            Id of the client
        :param client_name: str
            Name of the client
        :param training_params: list
            parameters for local training routine
        :param model: torch.nn.module
            Neural  network
        :param device: torch.device
            GPU  |  CPU
        :param log_wandb: bool
            Flag for logging results to wandb.ai
        """
        super(FedDisCollaborator, self).\
            __init__(client_idx=client_idx, client_name=client_name, training_params=training_params, model=model,
                     device=device, log_wandb=log_wandb)

        self.self_sup = training_params['self_supervision']['use']
        if self.self_sup:
            ses_params = training_params['self_supervision']['params']
            self.paint_rectangles = RandRectangleMasking(start_x_range=ses_params['start_x_range'],
                                                         start_y_range=ses_params['start_y_range'],
                                                         width_range=ses_params['width_range'],
                                                         max_rectangles=ses_params['max_rectangles'],
                                                         cval=ses_params['cval'])
        self.shift_ = RandAdjustContrast(prob=1.0, gamma=self.training_params['optimizer_params']['gamma_shift'])

        self.eval_step = 0
        self.min_val_loss = np.inf
        self.train_step = 0

        self.criterion_l1 = L1Loss().to(device)
        self.criterion_sl = CosineEmbeddingLoss().to(device)
        self.criterion_tv = TotalVariationLoss().to(device)


        self.lambda_R = self.training_params['optimizer_params']['lambda_R']
        self.lambda_S = self.training_params['optimizer_params']['lambda_S']
        self.lambda_L = self.training_params['optimizer_params']['lambda_L']
        self.loss_network = VGGEncoder().eval().to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=training_params['optimizer_params']['learning_rate'])
        self.decayRate = training_params['optimizer_params']['decay_rate']
        self.my_lr_scheduler = ExponentialLRDecay(optimizer=self.optimizer, gamma=self.decayRate)

    def load_data(self, dataset_name):
        """
        :param dataset_name: str or list
            name of the dataset(s) to load
        :return:
            local_training_data: monai.DataLoader
            local_test_data: monai.DataLoader
            num_train_samples: int
                number of training samples
        """
        dataset_name = [dataset_name] if isinstance(dataset_name, str) else dataset_name  # make list if input is str
        local_training_data, local_test_data, _ = self.data_loader.load_data(dataset_current_list=dataset_name)
        return local_training_data, local_test_data

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
        self.model.train().to(self.device)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        # self.optimizer.state = collections.defaultdict(dict) #  reset optimizer
        latent_loss = self.training_params['optimizer_params']['latent_loss']
        epoch_losses = []
        for epoch in range(self.training_params['nr_epochs']):
            start_time = time()
            batch_loss, batch_loss_l1, batch_loss_scl, batch_loss_lol = 1.0, 1.0, 1.0, 1.0
            count_images = 0

            for data in self.local_training_data:
                # Input
                input_all_slices = data[0]
                middle_slice, rand_slice = int(input_all_slices.shape[1] / 2), \
                                           np.random.randint(0, input_all_slices.shape[1], 1)[0]

                images = input_all_slices[:, np.newaxis, middle_slice, :, :]
                b, s, w, h = images.shape
                images = images.view(b*s, 1, h, w)

                images_cleaned = copy.deepcopy(images)
                images_masked = copy.deepcopy(images)

                # Intensity augmentation
                if self.training_params['optimizer_params']['use_shape_pair']:
                    # SCL w. second modality
                    intensity_augmented_images = \
                        data[1][:, np.newaxis, middle_slice, :, :].view(b*s, h, w).to(self.device)
                else:
                    # SCL w. gamma shift
                    intensity_augmented_images =\
                        torch.from_numpy(self.shift_([images_masked.cpu().numpy()])[0]).to(self.device)

                if self.self_sup:
                    # Self-supervision [In-Painting]
                    for b, img in enumerate(images_cleaned):
                        images_cleaned[b] = paint_anomalies(img)

                    # Self-supervision [Masking]
                    for b, img in enumerate(images_masked):
                        images_masked[b], intensity_augmented_images[b] = \
                            self.paint_rectangles([img, intensity_augmented_images[b]])

                count_images += images.shape[0]
                images, images_cleaned, images_masked \
                    = images.to(self.device), images_cleaned.to(self.device), images_masked.to(self.device)

                # Forward Pass
                self.optimizer.zero_grad()
                reconstructed_images, f_result = self.model(images_masked)

                # Reconstruction Loss
                loss_l1 = self.criterion_l1(reconstructed_images, images_cleaned)
                if latent_loss:
                    # Perceptual Loss
                    input_features = self.loss_network(images_cleaned.repeat(1, 3, 1, 1))
                    output_features = self.loss_network(reconstructed_images.repeat(1, 3, 1, 1))

                    l_scl = 0
                    for output_feature, input_feature in zip(output_features, input_features):
                        l_scl += self.criterion_l1(output_feature, input_feature)

                    # TV Loss
                    l_lol = self.criterion_tv(reconstructed_images)
                    # z_s, z_a = f_result['z_s'], f_result['z_a']
                    # z_s_fl, z_a_fl = torch.flatten(z_s, start_dim=1), torch.flatten(z_a, start_dim=1)
                    #
                    # # Get shape representations of augmented image
                    # with torch.no_grad():
                    #     self.test_model.load_state_dict(self.model.state_dict())
                    #     self.test_model.to(self.device).eval()
                    #     _, out_dict = self.test_model(intensity_augmented_images)  # shape of augmented appearance
                    #     z_s_shift = out_dict['z_s']
                    #     # Unit projection for shape latent space
                    #     m = torch.nn.Conv2d(z_s.shape[1], z_a.shape[1], 1, stride=1)
                    #     torch.nn.init.ones_(m.weight)
                    #     m = m.to(self.device)
                    #     z_s_ng = m(z_s)
                    #
                    # # Train appearance to be orthogonal to shared shape space
                    # l_lol = self.criterion_sl(z_a_fl, torch.flatten(z_s_ng.detach(), start_dim=1),
                    #                           torch.autograd.Variable(torch.Tensor(z_a.size(0)).cuda().fill_(-1.0)))
                    #
                    # # Train shape to be consistent under different intensity augmentations
                    # l_scl = self.criterion_sl(z_s_fl, torch.flatten(z_s_shift.detach(), start_dim=1),
                    #                           torch.autograd.Variable(torch.Tensor(z_a.size(0)).cuda().fill_(1.0)))

                    # Latent Consistency Loss
                    if round_idx >= self.training_params['optimizer_params']['round_scl_injection']:
                        loss = loss_l1 + self.lambda_R * (self.lambda_S * l_scl + self.lambda_L * l_lol)
                    else:
                        loss = loss_l1
                else:
                    loss = loss_l1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # to avoid nan loss
                self.optimizer.step()
                batch_loss += loss.item() * images.size(0)
                batch_loss_l1 += loss_l1.item() * images.size(0)
                if latent_loss:
                    batch_loss_lol += l_lol.item() * images.size(0)
                    batch_loss_scl += l_scl.item() * images.size(0)

                if count_images >= self.training_params['max_iterations']:
                    logging.info('Break due to maximal number of local iterations...')
                    break
            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_l1 = batch_loss_l1 / count_images if count_images > 0 else batch_loss_l1
            if latent_loss:
                epoch_loss_lol = batch_loss_lol / count_images if count_images > 0 else batch_loss_lol
                epoch_loss_scl = batch_loss_scl / count_images if count_images > 0 else batch_loss_scl

            epoch_losses.append(epoch_loss)
            end_time = time()
            wandb.log({"Train\Example_" + self.client_name: [
                wandb.Image(reconstructed_images.detach().cpu()[0].numpy(),
                            caption="Reconstruction_" + str(self.train_step))]})
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))

            wandb.log({"Train/Loss_" + self.client_name: epoch_loss, '_step_' + self.client_name: self.train_step})
            wandb.log(
                {"Train/Loss_l1_" + self.client_name: epoch_loss_l1, '_step_' + self.client_name: self.train_step})
            if latent_loss:
                wandb.log({"Train/Loss_LOL_" + self.client_name: epoch_loss_lol,
                           '_step_' + self.client_name: self.train_step})
                wandb.log({"Train/Loss_SCL_" + self.client_name: epoch_loss_scl,
                           '_step_' + self.client_name: self.train_step})

            torch.save(self.model.state_dict(), self.client_path + '/latest_model.pt')
            self.train_step += 1

            if epoch % 3 == 0:
                self.test(self.model.state_dict())

        return self.client_name, self.model.state_dict(), self.optimizer.state_dict(), \
               sum(epoch_losses) / len(epoch_losses)

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
        self.test_model.eval().to(self.device)
        metrics = {
            'test_loss_l1': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in self.local_test_data:
                x_all = data[0]
                middle_slice = int(x_all.shape[1] / 2)
                x = x_all[:, np.newaxis, middle_slice, :, :]
                images_cleaned = copy.deepcopy(x)
                images_masked = copy.deepcopy(x)

                if self.self_sup:
                    for b, img in enumerate(images_cleaned):
                        images_cleaned[b] = paint_anomalies(img)
                    for b, img in enumerate(images_masked):  # In-painting techniques
                        images_masked[b] = self.paint_rectangles([img])[0]

                test_total += x.shape[0]
                x, images_cleaned, images_masked = \
                    x.to(self.device), images_cleaned.to(self.device), images_masked.to(self.device)

                # Forward pass
                out_, _ = self.test_model(images_masked)
                y = out_
                loss_l1 = self.criterion_l1(y, images_cleaned)

                metrics['test_loss_l1'] += loss_l1.item() * x.size(0)

        wandb.log({"Test\Example_" + self.client_name: [wandb.Image(
            y.detach().cpu()[0].numpy(), caption="Reconstruction_" + str(self.train_step))]})

        for metric_key in metrics.keys():
            metric_name = 'Test/' + str(metric_key) + '_' + self.client_name
            metric_score = metrics[metric_key] / test_total if test_total != 0 else 0
            wandb.log({metric_name: metric_score, '_step_' + self.client_name: self.train_step})

        if loss_l1 < self.min_val_loss:
            self.min_val_loss = loss_l1
            torch.save(model_weights, self.client_path + '/best_model.pt')
        self.my_lr_scheduler.step()

        return metrics, test_total