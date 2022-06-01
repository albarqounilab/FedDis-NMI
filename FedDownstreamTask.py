"""
FedDownstreamTask.py

Run Downstream Tasks after training has finished

- Anomaly Detection

- Reconstruction Fidelity

"""
from monai.ILIA.core.FedDownstreamTask import FedDownstreamTask
from monai.losses import L1Loss
from monai.transforms import ToRGB, MedianFilter
from monai.metrics import DiceScore, PRCMetric, SSIMMetric, compute_meandice, \
    get_confusion_matrix, compute_confusion_matrix_metric

import torch
import logging
import numpy as np
import cv2
import copy

import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class FedDisDownstreamTask(FedDownstreamTask):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, model, device, params, test_data_dict, log_wandb):
        self.checkpoint_path = params['checkpoint_path']
        self.model = model.to(device)
        self.healthy_data = test_data_dict[0]
        self.anomaly_data = test_data_dict[1]
        self.test_data_dict = test_data_dict
        self.device = device
        self.criterion_l1 = L1Loss().to(self.device)
        self.ssim = SSIMMetric(data_range=1., reduction='none')
        self.median_filter = MedianFilter(median_kernel=3)
        self.dice_score = DiceScore()
        self.precision_recall_curve = PRCMetric()
        self.img2RBG = ToRGB(255, 255, 255)
        self.mask2RGB = ToRGB(255, 5, 159)
        super(FedDisDownstreamTask, self).__init__(model, device, params, test_data_dict, log_wandb)

    def start_task(self, global_models):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_models: dict
                   dictionary with the model weights of the federated collaborators
        """
        self.anomaly_test(global_models)
        self.reconstruction_fidelity(global_models)

    def anomaly_test(self, global_models):
        """
        Validation on all clients after a number of rounds
        Logs results to wandb

        :param model_global:
            Global parameters
        :param round_idx: int
            Round number
        """
        logging.info("################ ANOMALY TEST ON ALL CLIENTS #####################")
        client_metrics = dict()
        metrics = ['AUPRC', 'DICE']

        # VOLUME threshold computed in an unsupervised manner to achieve < 1% FPR on healthy data
        # ths =[Oasis, Adni-S, Adni-P, KRI] # Method
        # ths = [0.0681, 0.0671, 0.0721, 0.0631]  # Local
        # ths = [0.0831, 0.0701, 0.0691, 0.0541]  # FedAvg
        # ths = [0.0691, 0.0691, 0.0691, 0.0691]  # FedAvg
        # ths = [0.0871, 0.0501, 0.0791, 0.0450]  # SiloBN
        ths = [0.0731, 0.0691, 0.0521, 0.0350]    # FedLCL
        # ths = [0.0541, 0.0801, 0.0611, 0.0480]  # FedLOL
        # ths = [0.0731, 0.0691, 0.0511, 0.0340]  # FedSCL
        # ths = [0.0691, 0.0641, 0.0571, 0.0440]  # FedDis
        # ths = [0.0561, 0.0561, 0.0561, 0.0561]  # DC
        # ths = []
        for c_id, client_key in enumerate(global_models.keys()):
            th = ths[c_id]
            client_metrics[client_key] = dict()
            self.model.load_state_dict(global_models[client_key])
            self.model.eval()
            for metric in metrics:
                client_metrics[client_key][metric] = []
            for d_id, dataset_key in enumerate(self.anomaly_data.keys()):
                # th = ths[d_id]
                dataset = self.anomaly_data[dataset_key]
                test_metrics = dict()
                for metric in metrics:
                    test_metrics[metric] = []
                logging.info('DATASET: {}'.format(dataset_key))
                predictions, labels, save_labels, residuals, orig = [], [], [], [], []
                for idx, data in enumerate(dataset):
                    x, masks, brains = data[0].to(self.device), data[1], data[2]
                    nr_batches, nr_slices, width, height = x.shape
                    x_input = x.view(nr_batches * nr_slices, 1, width, height)

                    # Forward pass
                    x_rec, _ = self.model(x_input)

                    ##  !!!To compute simple post-processing on input image !!
                    # x_rec_dict = dict()
                    # x_rec_dict['x_rec'] = torch.zeros(x_input.shape)

                    x_rec = x_rec.view(nr_batches, nr_slices, width, height)
                    x, x_rec, masks, brains = x.cpu().detach().numpy(), x_rec.cpu().detach().numpy(), \
                                              masks.cpu().detach().numpy(), brains.cpu().detach().numpy()

                    # Post processing
                    diff = cv2.subtract(x, x_rec)  # Residual
                    diff[diff < 0] = 0  # Keep positive residual
                    diff_pp = self.median_filter(diff)   # Median Filter

                    if predictions == []:
                        predictions = diff_pp
                        labels = masks
                    else:
                        predictions = np.concatenate([predictions, diff_pp])
                        labels = np.concatenate([labels, masks])

                    diff_dice = copy.deepcopy(diff_pp)
                    # Binarization
                    diff_dice[diff_dice < th] = 0
                    diff_dice[diff_dice > 0] = 1
                    slice_dice, slice_mask = torch.from_numpy(diff_dice), torch.from_numpy(masks)
                    # Additional metrics
                    test_metrics['DICE'].append(self.dice_score(slice_dice, slice_mask))
                    orig.append(x)
                    residuals.append(diff)
                    save_labels.append(masks)

                    # Visual examples in wandb
                    if idx % 10 == 0:
                        auprc_slice, _, _, _ = self.precision_recall_curve(torch.from_numpy(diff_pp),
                                                                           torch.from_numpy(masks))
                        mid_slice = max(1, int(nr_slices / 2))
                        count_ = str(idx * nr_batches) + '-' + str(mid_slice)

                        img_color = self.img2RBG(x[0][mid_slice])
                        x_rec_color = self.img2RBG(x_rec[0][mid_slice])
                        mask_color = self.mask2RGB(masks[0][mid_slice].astype(bool))
                        grid_image = np.hstack([img_color, x_rec_color, mask_color])
                        wandb.log({"Anomaly_Reconstructions/" + client_key + '_' + dataset_key + '_' + str(count_) + '_' +
                                   str(mid_slice) + '_' + str(auprc_slice):
                                       [wandb.Image(grid_image, caption="Anomaly_" + str(count_))]})
                        plt.figure()
                        diffp = plt.imshow(diff_pp[0][mid_slice],  cmap='jet')
                        wandb.log({"Anomaly_Heatmaps/" + client_key + '_' + dataset_key + '_' + str(count_) + '_' +
                                   str(mid_slice) + '_' + str(auprc_slice):
                                       [wandb.Image(diffp, caption="Anomaly_" + str(count_))]})
                        plt.close()
                auprc, precisions, recalls, thresholds = self.precision_recall_curve(torch.from_numpy(predictions),
                                                                                     torch.from_numpy(labels))

                test_metrics['AUPRC'].append(auprc)

                for metric in test_metrics:
                    logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                    client_metrics[client_key][metric].append(test_metrics[metric])

                # Save results for further processing
                print(type(self.checkpoint_path))
                print(self.checkpoint_path)
                np.save(self.checkpoint_path + '/' + str(client_key) + ' ' + str(dataset_key) + '_orig.npy', np.asarray(orig))
                np.save(self.checkpoint_path + '/' + str(client_key) + '_' + str(dataset_key) + '_residuals.npy', np.asarray(residuals))
                np.save(self.checkpoint_path + '/' + str(client_key) + '_' + str(dataset_key) + '_labels.npy', np.asarray(save_labels))

        logging.info('Writing Box plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            for ck in client_metrics.keys():
                x = []
                y = []
                for idx, dataset_values in enumerate(client_metrics[ck][metric]):
                    logging.info('IDX: {}'.format(idx))
                    dataset_name = list(self.anomaly_data)[idx]
                    for dataset_val in dataset_values:
                        y.append(dataset_val)
                        x.append(dataset_name)

                fig_bp.add_trace(go.Box(
                    y=y,
                    x=x,
                    name=ck,
                    boxmean='sd'
                ))
            title = metric
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Anomaly_Metrics/" + str(metric): fig_bp})

    def reconstruction_fidelity(self, global_models):
        """
        Validation on all clients after a number of rounds
        Logs results to wandb

        :param model_global:
            Global parameters
        :param round_idx: int
            Round number
        """
        logging.info("################ RECONSTRUCTION TEST #################")
        client_metrics = dict()
        metrics = ['losses_l1', 'SSIM']

        for client_key in global_models.keys():
            self.model.load_state_dict(global_models[client_key])
            self.model.eval()
            client_metrics[client_key] = {
                    'losses_l1': [],
                    'SSIM': []
                }

            for dataset_key in self.healthy_data.keys():
                dataset = self.healthy_data[dataset_key]
                test_metrics = {
                    'losses_l1': [],
                    'SSIM': []
                }
                img_ct = -1
                logging.info('DATASET: {}'.format(dataset_key))
                for idx, data in enumerate(dataset):
                    img_ct += 1
                    x_all = data[0]
                    middle_slice = int(x_all.shape[1] / 2)
                    x = x_all[:, np.newaxis, middle_slice, :, :].to(self.device)
                    x_rec, _ = self.model(x)

                    for i in range(len(x)):
                        count = str(i * img_ct)
                        x_ = x[i][0]
                        x_rec_ = x_rec[i][0]
                        loss_l1 = self.criterion_l1(x_rec_, x_)

                        x_ = x_.cpu().detach()
                        x_rec_ = x_rec_.cpu().detach()
                        ssim_ = self.ssim(x_rec_, x_)

                        test_metrics['losses_l1'].append(loss_l1.item())
                        test_metrics['SSIM'].append(ssim_)

                for metric in test_metrics:
                    logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))

                    client_metrics[client_key][metric].append(test_metrics[metric])

        logging.info('Writing plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            for ck in client_metrics.keys():
                x = []
                y = []
                for idx, dataset_values in enumerate(client_metrics[ck][metric]):
                    dataset_name = list(self.healthy_data)[idx]
                    for dataset_val in dataset_values:
                        y.append(dataset_val)
                        x.append(dataset_name)

                fig_bp.add_trace(go.Box(
                    y=y,
                    x=x,
                    name=ck,
                    boxmean='sd'
                ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)/" + str(metric): fig_bp})
