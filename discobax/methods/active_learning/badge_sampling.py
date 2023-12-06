"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import scipy.stats
import numpy as np
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from discobax.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from slingpy import AbstractDataSource, AbstractBaseModel


class BadgeSampling(BaseBatchAcquisitionFunction):
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
        gradient_embedding = model.get_gradient_embedding(
            dataset_x.subset(available_indices)
        ).numpy()
        chosen = BadgeSampling.kmeans_initialise(
            gradient_embedding, acquisition_batch_size
        )
        selected = [available_indices[idx] for idx in chosen]
        return selected

    @staticmethod
    def kmeans_initialise(gradient_embedding, k):
        ind = np.argmax([np.linalg.norm(s, 2) for s in gradient_embedding])
        mu = [gradient_embedding[ind]]
        indsAll = [ind]
        centInds = [0.0] * len(gradient_embedding)
        cent = 0
        while len(mu) < k:
            if len(mu) == 1:
                D2 = pairwise_distances(gradient_embedding, mu).ravel().astype(float)
            else:
                newD = (
                    pairwise_distances(gradient_embedding, [mu[-1]])
                    .ravel()
                    .astype(float)
                )
                for i in range(len(gradient_embedding)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            D2 = D2.ravel().astype(float)
            Ddist = (D2**2) / sum(D2**2)
            customDist = scipy.stats.rv_discrete(
                name="custm", values=(np.arange(len(D2)), Ddist)
            )
            ind = customDist.rvs(size=1)[0]
            mu.append(gradient_embedding[ind])
            indsAll.append(ind)
            cent += 1
        gram = np.matmul(gradient_embedding[indsAll], gradient_embedding[indsAll].T)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        vgt = val[val > 1e-2]
        return indsAll
