from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import numpy as np

class ensembleEstimation():
    def __init__(self, max_depth, data):
        self.max_depth = max_depth
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']


    def approx_learner_dist(self):
        accuracies = []
        for _ in range(100):
            # Train a Random Forest with a single tree
            clf = RandomForestClassifier(
                        max_depth=self.max_depth,
                        n_estimators=1, 
                        random_state=0
                    )
            clf.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred = clf.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            accuracies.append(acc)
            del clf, y_pred
        
        accuracies = np.array(accuracies)
        self.mu_p = np.mean(accuracies)
        self.sigma_p = np.std(accuracies)

        return np.array(accuracies)


    def find_actual_accuracy(self, N):
        clf = RandomForestClassifier(
                    max_depth=self.max_depth,
                    n_estimators=N, 
                    random_state=1
                )
        clf.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
        del clf
    
        return acc
    

    def approximate(self, N, approx, type=None):
        def binom_approx(mu_p, N, type):
            prob = None
            if type == 'small':
                prob = norm.cdf((2 * mu_p - 1) * ((N + 1) / 2) ** 0.5)
            elif type == 'large':
                prob = norm.cdf((N * (2 * mu_p - 1) + 1) / (2 * (N * mu_p * (1 - mu_p)) ** 0.5))
            
            return prob

        def poisson_approx(mu_p, N):
            prob = norm.cdf((2 * mu_p - 1) * ((N + 1) / 2) ** 0.5)
            return prob

        def normal_approx(mu_p, sigma_p, N):
            prob = norm.cdf((N * (2 * mu_p - 1) + 1) \
                            / (2 * (N * (mu_p - sigma_p ** 2 - mu_p ** 2)) ** 0.5))
            return prob
        
        if approx == 'binomial':
            return binom_approx(mu_p=self.mu_p,
                                N=N,
                                type=type)
        elif approx == 'poisson':
            return poisson_approx(mu_p=self.mu_p,
                                    N=N)
        elif approx == 'normal':
            return normal_approx(mu_p=self.mu_p,
                                    sigma_p=self.sigma_p,
                                    N=N)
        return None