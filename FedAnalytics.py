"""
FedAnalytics.py

Run Analytics after an epoch/ training has finished

Nature MI includes running anomaly detection and reconstruction tests after the training has  finihsed

"""
import logging
from monai.ILIA.core.FedAnalytics import FedAnalytics

class FedDisAnalytics(FedAnalytics):
    """
    Federated Analytics
        - run analysis on epoch_end and on training_end
    """

    def on_epoch_end(self, global_models, loss_locals):
        """
        :param global_models: dict
            dictionary with the model weights of the federated collaborators
        :param loss_locals:
            the loss values of the epoch for all federated collaborators

        Default epoch stats are sufficient
        """
        logging.info('[FedAnalytics::on_epoch_end] Not implemented yet.')
        return

    def on_train_end(self, global_models):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, reconstruction fidelity tests

        :param global_models: dict
                   dictionary with the model weights of the federated collaborators
        """
        self.demographics_analysis(global_models)
        self.disease_severiy_analysis(global_models)
        self.statistical_analysis(global_models)

    def statistical_analysis(self, global_models):
        logging.info('[FedAnalytics::on_train_end::statistical_analysis] Not implemented yet.')
        return

    def demographics_analysis(self, global_models):
        logging.info('[FedAnalytics::on_train_end::demographics_analysis] Not implemented yet.')
        return

    def disease_severiy_analysis(self, global_models):
        logging.info('[FedAnalytics::on_train_end::disease_severiy_analysis] Not implemented yet.')
        return



