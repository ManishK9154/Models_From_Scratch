import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

class SimpleRegression:
    def __init__(self, learning_rate=0.01, iterations=100000, converge = 0.0001, alpha = 0.01, penalty = None):
        self.lr = learning_rate
        self.itr = iterations
        self.b = 0.0
        self.penalty = penalty
        self.alpha = alpha
        self.converge = converge
        self.w = 0.0

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        y = np.array(y).flatten()
        n = x.shape[0]
        last_mse = 0

        for _ in range(self.itr):
            y_cap = x@self.w + self.b
            err = y_cap - y
            err_sq = np.square(err)
            mse = np.sum(err_sq)/n
            print(f'Current MSE: {mse}')
            diff = mse-last_mse
            print(diff)
            dw = (2/n) * err @ x
            db = (2/n) * np.sum(err)

            if self.penalty=='l2':
                dw += self.alpha*self.w
            elif self.penalty=='l1':
                dw += self.alpha*np.sign(self.w)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.y_predicted = y_cap

            self.err_sq = np.sum(err_sq)/n
            self.residuals = err
            if np.abs(diff)<self.converge:
                break
            else:
                last_mse = mse
    def predict(self, x):
        return x @ self.w + self.b

class LogisticRegression():
    def __init__(self, learning_rate=0.01, iterations=100000, converge = 0.0001, alpha = 0.01, penalty = None):
        self.lr = learning_rate
        self.itr = iterations
        self.b = 0.0
        self.penalty = penalty
        self.alpha = alpha
        self.converge = converge
        self.w = 0.0
        self.loss = 0.0

    def model_accuracy(self, y_cap, y):
            df = pd.DataFrame({'y_cap': y_cap,'y': y})
            df['TP'] = 0
            df['FP'] = 0
            df['TN'] = 0
            df['FN'] = 0
            df.loc[(df['y_cap']==df['y']) & (df['y_cap']==1), 'TP'] = 1
            df.loc[(df['y_cap']!=df['y']) & (df['y_cap']==1), 'FP'] = 1
            df.loc[(df['y_cap']==df['y']) & (df['y_cap']==0), 'TN'] = 1
            df.loc[(df['y_cap']!=df['y']) & (df['y_cap']==0), 'FN'] = 1
            tp = df['TP'].sum()
            fp = df['FP'].sum()
            tn = df['TN'].sum()
            fn = df['FN'].sum()
            accuracy = (tp + tn)/(tp + tn + fp + fn)
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            f1 = 2 * (precision * recall)/(precision + recall)
            print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}')
            return accuracy, precision, recall, f1

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        # y = np.array(y).flatten()
        n = x.shape[0]
        last_loss = 0

        for _ in range(self.itr):
            y_cap = 1/(1 + np.exp(-(x@self.w + self.b)))
            loss = -(y * np.log(y_cap) + (1-y) * np.log(1-y_cap))/x.shape[0]
            y_class = [1 if x>=0.5 else 0 for x in y_cap]
            diff = loss.sum()-last_loss
            print(diff)
            dw = (1/n) * (y_cap-y)@x
            db = (1/n) * np.sum(y_cap-y)

            if self.penalty=='l2':
                dw += self.alpha*self.w
            elif self.penalty=='l1':
                dw += self.alpha*np.sign(self.w)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.y_class = y_class
            # self.model_accuracy(self.y_class, y)
            if np.abs(diff)<self.converge:
                break
            else:
                last_loss = loss.sum()
        self.loss = loss.sum()
        return self.model_accuracy(self.y_class, y)

    def predict_proba(self, x):
        y_cap = 1/(1 + np.exp(-(x@self.w + self.b)))
        return y_cap

    def predict_class(self,x):
        y_cap = 1/(1 + np.exp(-(x@self.w + self.b)))
        return [1 if x>=0.5 else 0 for x in y_cap]
 
    def loss_cal(self, y_cap, y):
        loss = -(y * np.log(y_cap) + (1-y) * np.log(1-y_cap))/len(y)
        final_loss = loss.sum()
        return 

    
          
    
class MinMaxScaler:
    def __init__(self):
        self.mn = 0.0
        self.mx = 0.0

    def fit(self, x):
            # x = np.array(x).flatten()
            self.mn = x.min(axis=0)
            self.mx = x.max(axis=0)
            x_new = (x-self.mn)/(self.mx-self.mn)
            return x_new

    def transform(self, x):
        return (x-self.mn)/(self.mx-self.mn)

class StandardScaler:
    def __init__(self):
        self.mu=0.0
        self.sigma=0.0

    def fit(self, x):
            # x = np.array(x).flatten()
            self.mu = np.mean(x, axis=0)
            self.sigma = np.std(x, axis=0)
            x_new = (x-self.mu)/self.sigma
            return x_new

    def transform(self, x):
        return (x-self.mu)/self.sigma

class RobustScaler:
    def __init__(self):
        self.md=0.0
        self.q1=0.0
        self.q3=0.0

    def fit(self, x):
            # x = np.array(x).flatten()
            self.md = np.median(x, axis=0)
            self.q1 = np.percentile(x, 25, axis=0)
            self.q3 = np.percentile(x, 75, axis=0)
            x_new = (x-self.md)/(self.q3-self.q1)
            return x_new

    def transform(self, x):
        return (x-self.md)/(self.q3-self.q1)

class TrainTestSplit:
     def __init__(self, split_pct = 0.3):
          self.split_pct = split_pct
          self.train_rows = 0.0

     def split(self, x: pd.DataFrame, y: pd.Series):
          self.train_rows = int(np.ceil(x.shape[0]*(1-self.split_pct)))
          rng = np.random.default_rng().choice(x.shape[0], size = self.train_rows, replace=False)
          rng = [int(x) for x in rng]
          mask = np.ones(x.shape[0], dtype=bool)
          mask[rng] = False
          return x.iloc[rng], y.iloc[rng], x.iloc[mask], y.iloc[mask]