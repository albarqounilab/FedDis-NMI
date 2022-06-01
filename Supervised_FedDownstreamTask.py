"""
Supervised_FedDownstreamTask.py

Run Downstream Tasks after training has finished

- Anomaly Detection

"""
from monai.ILIA.core.FedDownstreamTask import FedDownstreamTask
from monai.losses import L1Loss
from monai.transforms import ToRGB, MedianFilter
from monai.metrics import DiceScore, PRCMetric, SSIMMetric, get_confusion_matrix

import torch
import logging
import numpy as np
import copy

import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Supervised_FedDisDownstreamTask(FedDownstreamTask):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, model, device, params, test_data_dict, log_wandb):
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
        super(Supervised_FedDisDownstreamTask, self).__init__(model, device, params, test_data_dict, log_wandb)

    def start_task(self, global_models):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_models: dict
                   dictionary with the model weights of the federated collaborators
        """
        self.anomaly_test(global_models)

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

        for c_id, client_key in enumerate(global_models.keys()):
            client_metrics[client_key] = dict()
            self.model.load_state_dict(global_models[client_key])
            self.model.eval()
            for metric in metrics:
                client_metrics[client_key][metric] = []
            for d_id, dataset_key in enumerate(self.anomaly_data.keys()):
                dataset = self.anomaly_data[dataset_key]
                test_metrics = dict()
                for metric in metrics:
                    test_metrics[metric] = []
                predictions, labels, save_labels, residuals, orig = [], [], [], [], []
                ct_test = np.floor(int(0.7 * len(dataset)))
                ct = 0
                logging.info('DATASET: {}, with samples: {}'.format(dataset_key, ct_test))
                for idx, data in enumerate(dataset):
                    ct += 1
                    if ct <= ct_test:
                        continue
                    x, masks, brains = data[0].to(self.device), data[1], data[2]
                    nr_batches, nr_slices, width, height = x.shape

                    x_input = x.view(nr_batches * nr_slices, 1, width, height)
                    # Forward pass
                    x_rec, _ = self.model(x_input)
                    x_rec = x_rec.view(nr_batches, nr_slices, width, height)

                    x, x_rec, masks, brains = x.cpu().detach().numpy(), x_rec.cpu().detach().numpy(), \
                                              masks.cpu().detach().numpy(), brains.cpu().detach().numpy()
                    diff_pp = x_rec

                    if predictions == []:
                        predictions = diff_pp
                        labels = masks
                    else:
                        predictions = np.concatenate([predictions, diff_pp])
                        labels = np.concatenate([labels, masks])

                    diff_dice = copy.deepcopy(diff_pp)
                    # Binarization
                    diff_dice[diff_dice < 0.5] = 0
                    diff_dice[diff_dice > 0] = 1
                    slice_dice, slice_mask = torch.from_numpy(diff_dice), torch.from_numpy(
                        masks)
                    # Additional metrics
                    test_metrics['DICE'].append(self.dice_score(slice_dice, slice_mask))
                    orig.append(x)
                    residuals.append(diff_pp)
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
                        wandb.log(
                            {"Anomaly_Reconstructions/" + client_key + '_' + dataset_key + '_' + str(count_) + '_' +
                             str(mid_slice) + '_' + str(auprc_slice):
                                 [wandb.Image(grid_image, caption="Anomaly_" + str(count_))]})

                        plt.figure()
                        diffp = plt.imshow(diff_pp[0][mid_slice], cmap='jet')
                        wandb.log({"Anomaly_Heatmaps/" + client_key + '_' + dataset_key + '_' + str(count_) + '_' +
                                   str(mid_slice) + '_' + str(auprc_slice):
                                       [wandb.Image(diffp, caption="Anomaly_" + str(count_))]})
                        plt.close()
                auprc, precisions, recalls, thresholds = self.precision_recall_curve(torch.from_numpy(predictions),
                                                                                     torch.from_numpy(labels))
                test_metrics['AUPRC'].append(auprc)

                for metric in test_metrics:
                    client_metrics[client_key][metric].append(test_metrics[metric])

        logging.info('Writing Box plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            for ck in client_metrics.keys():
                x = []
                y = []
                for idx, dataset_values in enumerate(client_metrics[ck][metric]):
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
