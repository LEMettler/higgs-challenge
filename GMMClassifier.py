# (Bayesian) Gaussian Mixture Model for signal/background classification 
#
# Utilizing sklearns mixture models and calcuates the log-likelihood ratio. 
# A validation dataset can be used to optimize a critical value based on a given scorer function.
#
#
# Lukas Mettler - https://github.com/LEMettler

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class BGMClassifier:
    def __init__(self, n_components_signal=5, n_components_background=5):
        self.n_components_signal = n_components_signal
        self.n_components_background = n_components_background
    
    def train(self, x_train, y_train):
        """
        Trains two Bayesian GMMs for signal and background data.

        Parameters:
        - x_train: array-like of shape (n_samples, n_features)
            The input samples.
        - y_train: array-like of shape (n_samples,)
            The target values (binary labels: 0 for background, 1 for signal).
        - sample_weight: array-like of shape (n_samples,), optional
            The sample weights for training.
        """
        # Separate signal and background
        self.signal_data = x_train[y_train.astype(bool)]
        self.background_data = x_train[~y_train.astype(bool)]
        

        # Initialize and fit Bayesian GMs
        self.signal_bgm = BayesianGaussianMixture(n_components=self.n_components_signal, n_init=5, init_params='k-means++', random_state=1, max_iter=1000, tol=5e-4, warm_start=True)
        self.background_bgm = BayesianGaussianMixture(n_components=self.n_components_background, n_init=5, init_params='k-means++', random_state=1, max_iter=1000, tol=5e-4, warm_start=True)
        #init_params='random_from_data', 
        
        self.signal_bgm.fit(self.signal_data)
        print('Signal BGM calculated!')
        self.background_bgm.fit(self.background_data)
        print('Background BGM calculated!')

    def calculateLLR(self, x):
        """
        Calculate the log-likelihood ratios for the signal and background hypotheses.

        Parameters:
        - x: numpy array of shape (n_samples, n_features)

        Returns:
        - log_likelihood_ratio: numpy array of log-likelihood ratios
        """
        # Calculate log-likelihood for signal and background
        log_likelihood_signal = self.signal_bgm.score_samples(x)
        print('Signal Likelihood calculated!')
        log_likelihood_background = self.background_bgm.score_samples(x)
        print('Background Likelihood calculated!')

        # Compute the log-likelihood ratio
        log_likelihood_ratio = log_likelihood_signal - log_likelihood_background

        return log_likelihood_ratio
        
    def predict(self, x):
        """
        Predict the class labels (signal or background) for the given input data.

        Parameters:
        - x: numpy array of shape (n_samples, n_features), input data

        Returns:
        - probs: numpy array of predicted labels (0 for background, 1 for signal)
        """
        llr = self.calculateLLR(x)
        print(f'calculated llr')

        probs = np.zeros_like(llr)
        probs[llr > self.critical_value] = 1

        return probs

    def findCriticalValue(self, x_val, y_val, w_val, scorer_function, num_points=100):
        """
        Find the best critical value for the log-likelihood ratio using validation data.

        Parameters:
        - x_val: numpy array of shape (n_samples, n_features), validation data
        - y_val: numpy array of boolean values, true labels for validation data
        - w_val: numpy array of weights of y_val
        - scorer_function: function that evaluates the performance (e.g., ams_scorer)
        - num_points: number of points to evaluate for finding the critical value

        Returns:
        - best_critical_value: the critical value that maximizes the scorer function
        """
        # Calculate log-likelihood ratios for the validation set
        llr = self.calculateLLR(x_val)

        # Determine the range for searching critical values
        min_llr, max_llr = np.min(llr), np.max(llr)
        thresholds = np.linspace(min_llr, max_llr, num_points)

        best_score = -np.inf
        best_critical_value = None

        for threshold in thresholds:
            # Predict signal or background based on the threshold
            y_pred = llr > threshold
            # Calculate the score
            score = scorer_function(y_pred, y_val, w_val)

            if score > best_score:
                best_score = score
                best_critical_value = threshold

        print(f'Best critical value {best_critical_value} for validation score: {best_score}')

        self.critical_value = best_critical_value
        return best_critical_value, best_score

    def plotDistributions(self):
        """
        Plots the histograms of training signal and background data along with the fitted
        Bayesian Gaussian Mixture distributions for each feature. Signal plots
        are on the left, background plots are on the right.
        """
        def gaussian(x, mean, covariance):
            """Return the value of a Gaussian distribution with given mean and covariance."""
            return np.exp(-0.5 * ((x - mean) ** 2) / covariance) / np.sqrt(2 * np.pi * covariance)
        
        def plotHistogramAndFits(ax, label, data, color, weights, means, covs):
            """Plot histogram and Gaussian mixture components."""
            
            ax.hist(data, bins=50, density=True, alpha=0.5, color=color)
            x_range = np.linspace(data.min(), data.max(), 500)
            
            total_mixture = 0
            for weight, mean, cov in zip(weights, means, covs):
                this_gauss = weight * gaussian(x_range, mean, cov)
                total_mixture += this_gauss
                ax.plot(x_range, this_gauss, color=color, linestyle='--')
            ax.plot(x_range, total_mixture, color='k', linestyle='-')
            
            ax.set_xlim(data.min(), data.max())
            ax.set_title(label)

        signal_data = self.signal_data
        background_data = self.background_data
        n_features = signal_data.shape[1]
        n_rows = (n_features + 1) // 2  # Two features per row

        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

        for i in range(n_features):
            row = i // 2
            col_signal = 2 * (i % 2)
            col_background = col_signal + 1

            # Plot for signal
            ax_signal = axes[row, col_signal]
            plotHistogramAndFits(ax_signal, f'Feature {i + 1} - Signal', 
                                 signal_data[:, i], 'blue', 
                                 self.signal_bgm.weights_, 
                                 self.signal_bgm.means_[:, i], 
                                 self.signal_bgm.covariances_[:, i, i])

            # Plot for background
            ax_background = axes[row, col_background]
            plotHistogramAndFits(ax_background, f'Feature {i + 1} - Background', 
                                 background_data[:, i], 'red', 
                                 self.background_bgm.weights_, 
                                 self.background_bgm.means_[:, i], 
                                 self.background_bgm.covariances_[:, i, i])

        plt.tight_layout()
        plt.show()
