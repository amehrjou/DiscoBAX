"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import numpy as np
import torch
import random
from typing import List, AnyStr
from discobax.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from slingpy import AbstractDataSource, AbstractBaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdversarialBIM(BaseBatchAcquisitionFunction):
    def __init__(self, args={}):
        # Get hyperparameters from args dict
        if "eps" in args:
            self.eps = args["eps"]
        else:
            self.eps = 0.05

        if "verbose" in args:
            self.verbose = args["verbose"]
        else:
            self.verbose = False

        if "stop_iterations_by_count" in args:
            self.stop_iterations_by_count = args["stop_iterations_by_count"]
        else:
            self.stop_iterations_by_count = 1000

        if "gamma" in args:
            self.gamma = args["gamma"]
        else:
            self.gamma = 0.35

        if "adversarial_sample_ratio" in args:
            self.adversarial_sample_ratio = args["adversarial_sample_ratio"]
        else:
            self.adversarial_sample_ratio = 0.1

        super(AdversarialBIM, self).__init__()

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model",
    ) -> List:
        model.model.model = model.model.model.to(device)
        selected_portion = np.random.choice(
            available_indices,
            size=int(len(available_indices) * self.adversarial_sample_ratio),
            replace=False,
        )
        dis = np.zeros(len(available_indices)) + np.inf
        data_pool = dataset_x.subset(available_indices)

        random.shuffle(available_indices)
        for i, index in enumerate(available_indices[:1000]):
            if self.verbose:
                if i % 100 == 0:
                    print("adv {}/{}".format(i, len(available_indices)))
            x = torch.as_tensor(
                data_pool.subset([index]).get_data()[0].reshape(1, 1, -1)
            ).to(device)
            dis[i] = self.cal_dis(x, model)

        chosen = dis.argsort()[:acquisition_batch_size]
        for x in np.sort(dis)[:acquisition_batch_size]:
            print(x)
        return [available_indices[idx] for idx in chosen]

    def cal_dis(self, x, model):
        nx = x.detach()
        first_x = torch.clone(nx)

        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to(device)
        iteration = 0

        while torch.linalg.norm(nx + eta - first_x) < self.gamma * torch.linalg.norm(
            first_x
        ):
            if iteration >= self.stop_iterations_by_count:
                break

            out = torch.as_tensor(
                model.get_model_prediction(nx + eta, return_multiple_preds=True)[0]
            )
            out = torch.squeeze(out)
            variance = torch.var(out)
            variance.backward()

            # print("variance is {}".format(variance))

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            iteration += 1

        return (eta * eta).sum()
