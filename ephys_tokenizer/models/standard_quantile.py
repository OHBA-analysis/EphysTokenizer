"""Implementation of the standard-quantile tokenizer model."""

# Import packages
import logging
import numpy as np
import os
import pickle
from pqdm.threads import pqdm
from tqdm.auto import trange
from typing import Union, List, Optional
from ephys_tokenizer.configs import Config, get_config


_logger = logging.getLogger(__name__)


class StandardQuantileTokenizer:
    """
    StandardQuantileTokenizer class.

    Note that this class is framework-agnostic and can be used with
    any deep learning framework.

    This tokenizer employs the standard scaling and uniform binning
    quantisation, as discussed in the Chronos paper (Ansari et al., 2024).

    Parameters
    ----------
    config : Config
        Configuration object.
    """
    def __init__(self, config: Config):
        self.base_config = config
        self.config = config.config_class
        self.vocab = {}
        self.n_tokens = self.config.n_tokens
        self.standardize = self.config.standardize

    def fit(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        clip: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Fits the tokenizer to the data.

        Parameters
        ----------
        x : Union[np.ndarray, List[np.ndarray]]
            Input data to fit the tokenizer on.
        clip : Optional[Union[int, float]]
            Value to clip the input data to.
        """
        if not isinstance(x, list):
            x = [x]

        if self.standardize:
            for i in range(len(x)):
                x[i] = self._standardize(x[i])

        if clip is not None:
            x = [np.clip(x_i, a_min=-clip, a_max=clip) for x_i in x]

        self.vocab["bins"] = self.get_bins(x)
        self.vocab["bins_average"] = self.get_bins_average(self.vocab["bins"])
        self.vocab["total_token_counts"] = self.get_token_counts(x)

    def get_token_counts(
        self, data: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Gets the token counts for the given data.

        Parameters
        ----------
        data : Union[np.ndarray, List[np.ndarray]]
            The data to get token counts for.

        Returns
        -------
        token_counts : np.ndarray
            The token counts.
        """
        tokens = self.tokenize_data(data, concatenate=True)
        token_counts = np.bincount(
            tokens.flatten(), minlength=self.n_tokens
        )
        return token_counts

    # -------------------------
    # Standardization & Binning
    # -------------------------

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """
        Standardizes the data.

        Note
        ----
        The standard scaling methods are applied on the individual
        time series.
        """
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)
        return x

    def get_bins(
        self, data: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Gets the bins for the tokenizer.

        Parameters
        ----------
        data : Union[np.ndarray, List[np.ndarray]]
            The data to get bin edges.

        Returns
        -------
        bins : np.ndarray
            The bin edges.
        """
        if not isinstance(data, list):
            data = [data]
        data = np.concatenate(data, axis=0)
        data = data.flatten()

        # Apply quantile binning to get bin edges
        bins = np.quantile(
            data,
            q=np.linspace(0, 1, self.n_tokens - 1),
            method="inverted_cdf",  # closest to previous tensorflow version
        )  # bins.shape = (n_tokens - 1,)
        # NOTE: The tokens 0 and n_tokens - 1 are for values <= bins[0] and 
        #       >= bins[-1].
        #       The rest n_tokens - 2 are for values in between the bins are 
        #       equally spaced in the range of (bins[0], bins[-1]).

        # Match data types with MuTransformTokenizer bins
        bins = np.asarray(bins).astype(np.float64).copy()

        # Add epsilon to avoid non-inclusive bin edges
        bins[0] += 1e-6
        bins[-1] -= 1e-6
        # NOTE: Ensures the first and last bins are not empty.

        return bins

    def get_bins_average(self, bins: np.ndarray) -> np.ndarray:
        """
        Gets the bin centers for the tokenizer.

        Parameters
        ----------
        bins : np.ndarray
            The bin edges.

        Returns
        -------
        bins_average : np.ndarray
            The bin centers.
        """
        # Calculate the average of each bin
        bins_average = np.zeros(len(bins) + 1)
        bins_average[0] = bins[0]
        bins_average[-1] = bins[-1]

        for i in range(len(bins) - 1):
            bins_average[i + 1] = (bins[i] + bins[i + 1]) / 2
        return bins_average

    # -----------------------------
    # Tokenization & Reconstruction
    # -----------------------------

    def tokenize_data(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        concatenate: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Tokenizes the input data into discrete bins.

        Parameters
        ----------
        data : Union[np.ndarray, List[np.ndarray]]
            The input data to tokenize.
        concatenate : bool, optional
            Whether to concatenate the tokenized data into a single array.
        n_jobs : int, optional
            The number of jobs to run in parallel.

        Returns
        -------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokenized data.
        """
        if not isinstance(data, list):
            data = [data]

        def _tokenize_data_per_session(d):
            t = np.digitize(d, self.vocab["bins"])
            t = np.clip(t, 0, self.n_tokens - 1)  # safety check
            return t

        # Run tokenization
        kwargs = [{"d": d} for d in data]
        if len(data) == 1:
            _logger.info("Tokenizing data...")
            tokens = [_tokenize_data_per_session(**kwargs[0])]

        elif n_jobs == 1:
            tokens = []
            for i in trange(len(data), desc="Tokenizing data"):
                tokens.append(_tokenize_data_per_session(**kwargs[i]))

        else:
            tokens = pqdm(
                kwargs,
                _tokenize_data_per_session,
                n_jobs=n_jobs,
                desc="Tokenizing data",
                argument_type="kwargs",
                exception_behaviour="immediate",
            )

        if concatenate or len(tokens) == 1:
            tokens = np.concatenate(tokens)
        return tokens

    def reconstruct_data(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        concatenate: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reconstructs the original data from the quantized representation.

        Parameters
        ----------
        tokens : Union[np.ndarray, List[np.ndarray]]
            The tokenized data to reconstruct.
        concatenate : bool, optional
            Whether to concatenate the reconstructed data into a single array.
        n_jobs : int, optional
            The number of jobs to run in parallel.

        Returns
        -------
        reconstructed_data : Union[np.ndarray, List[np.ndarray]]
            The reconstructed data.
        """
        if not isinstance(tokens, list):
            tokens = [tokens]

        def _reconstruct_data_per_session(t):
            t = np.clip(t, 0, len(self.vocab["bins_average"]) - 1)  # safety check
            x = self.vocab["bins_average"][t]
            return x

        # Keywords for parallel processing
        kwargs = [{"t": t} for t in tokens]
        if len(tokens) == 1:
            _logger.info("Reconstructing data...")
            reconstructed_data = [_reconstruct_data_per_session(**kwargs[0])]

        elif n_jobs == 1:
            reconstructed_data = []
            for i in trange(len(tokens), desc="Reconstructing data"):
                reconstructed_data.append(_reconstruct_data_per_session(**kwargs[i]))

        else:
            reconstructed_data = pqdm(
                kwargs,
                _reconstruct_data_per_session,
                n_jobs=n_jobs,
                desc="Reconstructing data",
                argument_type="kwargs",
                exception_behaviour="immediate",
            )

        if concatenate or len(reconstructed_data) == 1:
            reconstructed_data = np.concatenate(reconstructed_data)
        return reconstructed_data

    # -----------------
    # Post hoc analysis
    # -----------------

    def get_pve(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        n_jobs: Optional[int] = 1,
    ) -> np.ndarray:
        """
        Computes the percentage of variance explained by the tokens.

        Parameters
        ----------
        data : Union[np.ndarray, List[np.ndarray]]
            Input time series data.
            Each element should be of shape (n_samples, n_channels).
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1.

        Returns
        -------
        pve : np.ndarray
            The percentage of variance explained by the tokens for each subject/session.
        """
        if not isinstance(data, list):
            data = [data]

        tokens = self.tokenize_data(data, n_jobs=n_jobs)
        reconstructed_data = self.reconstruct_data(tokens, n_jobs=n_jobs)

        if not isinstance(reconstructed_data, list):
            reconstructed_data = [reconstructed_data]

        pve = []
        for i in range(len(data)):
            original_x = data[i]
            reconstructed_x = reconstructed_data[i]
            pve.append(
                100 * (1 - np.sum((original_x - reconstructed_x) ** 2) / np.sum(original_x ** 2))
            )

        if len(pve) == 1:
            return pve[0]
        return np.array(pve)

    # ----------------
    # Saving & Loading
    # ----------------

    def save(self, dirname: str) -> None:
        """
        Saves the token vocabulary.

        Parameters
        ----------
        dirname : str
            Directory to save the model.
        """
        os.makedirs(dirname, exist_ok=True)

        # Save token vocabulary
        with open(f"{dirname}/vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        _logger.info(f"Saved token vocabulary in {dirname}.")

    @staticmethod
    def load_config(dirname: str) -> Config:
        """
        Loads the config from a directory.

        Parameters
        ----------
        dirname : str
            Directory to load the configuration from.

        Returns
        -------
        config : Config
            Configuration object.
        """
        return get_config(f"{dirname}/config.yml")

    @classmethod
    def load_model(cls, dirname: str):
        """
        Loads a saved model.

        Parameters
        ----------
        dirname : str
            Directory containing the saved model.

        Returns
        -------
        model : StandardQuantileTokenizer
            The loaded model.
        """
        config = cls.load_config(dirname)
        model = cls(config)
        with open(f"{dirname}/vocab.pkl", "rb") as f:
            model.vocab = pickle.load(f)
        return model
