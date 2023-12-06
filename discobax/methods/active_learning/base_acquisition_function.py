from abc import abstractmethod
from typing import AnyStr, List
from slingpy import AbstractDataSource, AbstractBaseModel


class BaseBatchAcquisitionFunction(object):
    @abstractmethod
    def __call__(self,
                dataset_x: AbstractDataSource,
                acquisition_batch_size: int,
                available_indices: List[AnyStr],
                last_selected_indices: List[AnyStr], 
                cumulative_indices: List[AnyStr] = None,
                model: AbstractBaseModel = None,
                dataset_y: AbstractDataSource = None,
                temp_folder_name: str = 'tmp/model',
                ) -> List:
        """
        Nominate experiments for the next learning round.

        Args:
            dataset_x: The dataset containing all training samples.
            batch_size: Size of the batch to acquire.
            available_indices: The list of the indices (names) of the samples not
                chosen in the previous rounds.
            last_selected_indices: The set of indices selected in the prrevious
                cycle.
            last_model: The prediction model trained by labeled samples chosen so far.

        Returns:
            A list of indices (names) of the samples chosen for the next round.
        """
        raise NotImplementedError()