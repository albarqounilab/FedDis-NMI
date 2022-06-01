"""
FedPlanner.py

FedDis class for planning federated experiments
    - overrides load_data()

"""
from monai.ILIA.core.FedPlanner import FedPlanner
from monai.ILIA.core.utils import *


class FedDisFedPlanner(FedPlanner):
    """
    Federated Planner
        - parametrization of the federated training via 'config.yaml'
        - initializes FedCollaborator:
            clients with an id, selection probability (and their local data-sets - for simulations)
        - initializes FedAnalytics
        - initializes FedVisu
        - initializes FedDownstreamTask
        - starts the federated experiments -> FedAggregator
    """
    def load_data(self, test_data_params):
        """
        :param dataset_name: str
            name of the dataset(s) to load
        :return: list of
           test_data_dict:
            test_data_dict[dataset_name] = dataset
        """
        healthy_data_module = import_module(test_data_params['data_loader_healthy']['module_name'],
                                                 test_data_params['data_loader_healthy']['class_name'])
        healthy_loader = healthy_data_module(**(test_data_params['data_loader_healthy']['params']))
        anomaly_data_module = import_module(test_data_params['data_loader_anomaly']['module_name'],
                                                 test_data_params['data_loader_anomaly']['class_name'])
        anomaly_loader = anomaly_data_module(**(test_data_params['data_loader_anomaly']['params']))

        healthy_dataset_names = test_data_params['data_loader_healthy']['dataset_names']
        anomaly_dataset_names = test_data_params['data_loader_anomaly']['dataset_names']
        healthy_datasets = dict()
        for dataset_name in healthy_dataset_names:
            _, _, test_data = healthy_loader.load_data([dataset_name])
            healthy_datasets[dataset_name] = test_data

        anomaly_datasets = dict()
        for dataset_name in anomaly_dataset_names:
            anomaly_datasets[dataset_name] = anomaly_loader.load_data([dataset_name])

        return [healthy_datasets, anomaly_datasets]
