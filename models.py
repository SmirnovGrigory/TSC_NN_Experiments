import multiprocessing

from sktime.classification.hybrid import HIVECOTEV1

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.interval_based import (
    RandomIntervalSpectralEnsemble,
    TimeSeriesForestClassifier,
)
from sktime.classification.shapelet_based import ShapeletTransformClassifier

import torch
import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time

import os
from multiprocessing import Process
from multiprocessing.pool import Pool

class TSCDataset_MY(Dataset):
    def __init__(self, X):
        self.x = X

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor([self.x[i]]).float()

def parallelFitWrapper(X, y, model, name):
    model.fit(X,y)
    model.save("./" + name)

class HIVECOTEV1_MY(HIVECOTEV1):
    _tags = {
        "capability:multithreading": True,
        "classifier_type": "hybrid",
    }

    def __init__(
            self,
            stc_params=None,
            tsf_params=None,
            rise_params=None,
            cboss_params=None,
            verbose=0,
            n_jobs=1,
            random_state=None,
    ):
        self.stc_params = stc_params
        self.tsf_params = tsf_params
        self.rise_params = rise_params
        self.cboss_params = cboss_params

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.stc_weight_ = 0
        self.tsf_weight_ = 0
        self.rise_weight_ = 0
        self.cboss_weight_ = 0

        self.cnn_weight_ = 0

        self._stc_params = stc_params
        self._tsf_params = tsf_params
        self._rise_params = rise_params
        self._cboss_params = cboss_params
        self._stc = None
        self._tsf = None
        self._rise = None
        self._cboss = None

        self.cnn = None

        super(HIVECOTEV1, self).__init__()

    def _fit(self, X, y):
        """Fit HIVE-COTE 1.0 to training data.
        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.
        Returns
        -------
        self :
            Reference to self.
        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        # Default values from HC1 paper
        print("STARTED")
        if self.stc_params is None:
            self._stc_params = {"transform_limit_in_minutes": 120}
        if self.tsf_params is None:
            self._tsf_params = {"n_estimators": 500}
        if self.rise_params is None:
            self._rise_params = {"n_estimators": 500}
        if self.cboss_params is None:
            self._cboss_params = {}

        # Cross-validation size for TSF and RISE
        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class

        # Build STC
        self._stc = ShapeletTransformClassifier(
            **self._stc_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        #self._stc.fit(X, y)

        # Build TSF
        self._tsf = TimeSeriesForestClassifier(
            **self._tsf_params,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        #self._tsf.fit(X, y)

        # Build RISE
        self._rise = RandomIntervalSpectralEnsemble(
            **self._rise_params,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        print("RISE training")
        self._rise.fit(X, y)

        # Build cBOSS
        self._cboss = ContractableBOSS(
            **self._cboss_params,
            random_state=self.random_state,
            n_jobs=self._threads_to_use,
        )
        #self._cboss.fit(X, y)
        
        # proc_stc = Process(target=parallelFitWrapper, args=(X, y, self._stc, "stc"))
        # proc_tsf = Process(target=parallelFitWrapper, args=(X, y, self._tsf, "tsf"))
        # proc_rise = Process(target=parallelFitWrapper, args=(X, y, self._rise, "rise"))
        # proc_cboss = Process(target=parallelFitWrapper, args=(X, y, self._cboss, "cboss"))
        # procs = [proc_stc, proc_tsf, proc_rise, proc_cboss]
        # print("TRAINING IS STARTED")
        # for proc in procs:
        #     proc.start()
        #
        # for proc in procs:
        #     proc.join()

        with Pool(processes=4) as pool:
            print("parrallel training")
            proc_stc = pool.apply_async(parallelFitWrapper, (X, y, self._stc, "stc"))
            proc_tsf = pool.apply_async(parallelFitWrapper, (X, y, self._tsf, "tsf"))
            #proc_rise = pool.apply_async(parallelFitWrapper, (X, y, self._rise, "rise"))
            proc_cboss = pool.apply_async(parallelFitWrapper, (X, y, self._cboss, "cboss"))
            procs = [proc_stc, proc_tsf, proc_cboss]
            for proc in procs:
                proc.get()

        self._stc = self._stc.load_from_path("/home/ironic/repos/TSC_NN_Experiments/stc.zip")
        self._tsf =self._tsf.load_from_path("/home/ironic/repos/TSC_NN_Experiments/tsf.zip")
        #self._rise = self._rise.load_from_path("/home/ironic/repos/TSC_NN_Experiments/rise.zip")
        self._cboss = self._cboss.load_from_path("/home/ironic/repos/TSC_NN_Experiments/cboss.zip")

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find STC weight using train set estimate
        train_probs = self._stc._get_train_probs(X, y)
        train_preds = self._stc.classes_[np.argmax(train_probs, axis=1)]
        self.stc_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "STC train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("STC weight = " + str(self.stc_weight_))  # noqa

        if self.verbose > 0:
            print("TSF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find TSF weight using train set estimate found through CV
        train_preds = cross_val_predict(
            TimeSeriesForestClassifier(
                **self._tsf_params, random_state=self.random_state
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self._threads_to_use,
        )
        self.tsf_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "TSF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("TSF weight = " + str(self.tsf_weight_))  # noqa

        if self.verbose > 0:
            print("RISE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        # Find RISE weight using train set estimate found through CV
        train_preds = cross_val_predict(
            RandomIntervalSpectralEnsemble(
                **self._rise_params,
                random_state=self.random_state,
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self._threads_to_use,
        )
        self.rise_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "RISE train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("RISE weight = " + str(self.rise_weight_))  # noqa

        # Find cBOSS weight using train set estimate
        train_probs = self._cboss._get_train_probs(X, y)
        train_preds = self._cboss.classes_[np.argmax(train_probs, axis=1)]
        self.cboss_weight_ = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "cBOSS (estimate included)",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("cBOSS weight = " + str(self.cboss_weight_))  # noqa

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.
        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.
        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_cnn(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.
        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.
        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba_cnn(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.
        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.
        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))

        # Call predict proba on each classifier, multiply the probabilities by the
        # classifiers weight then add them to the current HC1 probabilities
        dists = np.add(
            dists,
            self._stc.predict_proba(X) * (np.ones(self.n_classes_) * self.stc_weight_),
        )
        dists = np.add(
            dists,
            self._tsf.predict_proba(X) * (np.ones(self.n_classes_) * self.tsf_weight_),
        )
        dists = np.add(
            dists,
            self._rise.predict_proba(X)
            * (np.ones(self.n_classes_) * self.rise_weight_),
        )
        dists = np.add(
            dists,
            self._cboss.predict_proba(X)
            * (np.ones(self.n_classes_) * self.cboss_weight_),
        )

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

    def _predict_proba_cnn(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.
        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.
        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))

        # Call predict proba on each classifier, multiply the probabilities by the
        # classifiers weight then add them to the current HC1 probabilities
        dists = np.add(
            dists,
            self._stc.predict_proba(X) * (np.ones(self.n_classes_) * self.stc_weight_),
        )
        dists = np.add(
            dists,
            self._tsf.predict_proba(X) * (np.ones(self.n_classes_) * self.tsf_weight_),
        )
        dists = np.add(
            dists,
            self._rise.predict_proba(X)
            * (np.ones(self.n_classes_) * self.rise_weight_),
        )
        dists = np.add(
            dists,
            self._cboss.predict_proba(X)
            * (np.ones(self.n_classes_) * self.cboss_weight_),
        )
        dataset_MY = TSCDataset_MY(X)
        dataloader_MY = DataLoader(dataset_MY, batch_size=dataset_MY.__len__(), shuffle=False)
        softmax = nn.Softmax()
        for _, X_MY in enumerate(dataloader_MY):
            cnn_pred = softmax(self.cnn(X_MY)).detach().numpy()
        dists = np.add(
            dists,
            cnn_pred
            * (np.ones(self.n_classes_) * self.cnn_weight_),
        )

        # Make each instances probability array sum to 1 and return
        return dists / dists.sum(axis=1, keepdims=True)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.
        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.
        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sklearn.ensemble import RandomForestClassifier

        if parameter_set == "results_comparison":
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                "tsf_params": {"n_estimators": 3},
                "rise_params": {"n_estimators": 3},
                "cboss_params": {"n_parameter_samples": 5, "max_ensemble_size": 3},
            }
        else:
            return {
                "stc_params": {
                    "estimator": RandomForestClassifier(n_estimators=1),
                    "n_shapelet_samples": 5,
                    "max_shapelets": 5,
                    "batch_size": 5,
                },
                "tsf_params": {"n_estimators": 1},
                "rise_params": {"n_estimators": 1},
                "cboss_params": {"n_parameter_samples": 1, "max_ensemble_size": 1},
            }


class TimeSeriesCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(5888, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
            # nn.Softmax()
        )

    def forward(self, xb):
        data = self.conv(xb)
        data = torch.flatten(data, start_dim=1)
        return self.linear(data)


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #print(x.size(0))
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        data = self.conv(x)
        #print(data.shape)
        #data = torch.flatten(data, start_dim=1)
        #print(data.shape)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(data, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out
