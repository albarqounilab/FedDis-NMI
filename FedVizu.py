"""
FedVizu.py

Run Vizualization after an epoch/ training has finished

"""
import logging
from monai.ILIA.core.FedVizu import FedVizu


class FedDisVizu(FedVizu):
    """
    Federated Vizualization
        - run vizualization on epoch_end and on training_end, e.g. UMAP plot, heat-maps, gradients, etc..
    """

    def on_epoch_end(self, global_models, loss_locals):
        """
        :param global_models: dict
            dictionary with the model weights of the federated collaborators
        :param loss_locals:
            the loss values of the epoch for all federated collaborators

        The function does not return anything. Use std output or files for printing and plotting statistics
        """

        logging.info('[FedVizu::on_epoch_end]: Not implemented yet.')
        return

    def on_train_end(self, global_models):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_models: dict
                   dictionary with the model weights of the federated collaborators
        """
        logging.info('[FedVizu::on_train_end]: Not implemented yet.')
        return
