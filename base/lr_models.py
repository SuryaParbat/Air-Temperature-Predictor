"""
We are going to create different Linear regression models.
"""

import logging, os, pickle
import pandas as pd
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from create_new_directory import create_directory
from logger import moniter
import numpy as np


class LinearRegressionModels:

    def __init__(self):
        pass

    @moniter
    def lr_model(self, x_train, x_test, y_train, y_test):

        try:
            lin_reg = LinearRegression()
            lin_reg.fit(x_train, y_train)
            # Calculating Accuracy
            lin_reg_accuracy = lin_reg.score(x_test, y_test)

            logging.debug(f"""lr_model is Successfully loaded\n""")
            return lin_reg_accuracy, lin_reg

        except Exception as e:
            logging.error(f""" ERROR IN : lr_model : {str(e)}\n""")
            return f"""ERROR IN : lr_model : {str(e)}\n"""

    @moniter
    def lasso_model(self, x_train, x_test, y_train, y_test):

        try:
            lassocv = LassoCV(alphas=None, cv=50, max_iter=20000, normalize=True)
            lassocv.fit(x_train, y_train)
            lasso = Lasso(alpha=lassocv.alpha_)
            lasso.fit(x_train, y_train)
            lasso_model_accuracy = lasso.score(x_test,y_test)
            logging.debug(f"""lasso_model is Successfully executed\n""")
            return lasso_model_accuracy, lasso
        except Exception as e:
            logging.error(f""" ERROR IN : lasso_model : {str(e)}\n""")
            return f"""ERROR IN : lasso_model : {str(e)}\n"""

    @moniter
    def ridge_model(self, x_train, x_test, y_train, y_test):

        try:
            ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=10, normalize=True)
            ridgecv.fit(x_train, y_train)
            ridge = Ridge(alpha=ridgecv.alpha_)
            ridge.fit(x_train, y_train)
            ridge_model_accuracy = ridge.score(x_test,y_test)
            logging.debug(f"""ridge_model is Successfully executed\n""")
            return ridge_model_accuracy, ridge
        except Exception as e:
            logging.error(f""" ERROR IN : ridge_model : {str(e)}\n""")
            return f"""ERROR IN : ridge_model : {str(e)}\n"""

    @moniter
    def elastic_net_model(self, x_train, x_test, y_train, y_test):

        try:
            elastic = ElasticNetCV(alphas=None, cv=10)
            elastic.fit(x_train, y_train)
            elastic_lr = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio_)
            elastic_lr.fit(x_train, y_train)
            elactic_model_accuracy = elastic_lr.score(x_test,y_test)
            logging.debug(f"""elastic_net_model is Successfully executed\n""")
            return elactic_model_accuracy, elastic_lr
        except Exception as e:
            logging.error(f""" ERROR IN : elastic_net_model : {str(e)}\n""")
            return f"""ERROR IN : elastic_net_model : {str(e)}\n"""

    # Let's create a function to create adjusted R-Squared
    @moniter
    def adj_r2(self,x, y, model):

        try:
            r2 = model.score(x, y)
            n = x.shape[0]
            p = x.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            logging.debug(f"""adj_r2 is Successfully executed\n""")
            return adjusted_r2
        except Exception as e:
            logging.error(f""" ERROR IN : adj_r2 : {str(e)}\n""")
            return f"""ERROR IN : adj_r2 : {str(e)}\n"""

    @moniter
    def create_pickle_file(self, model):

        try:
            '''
            # Now creating pickle file for this model
            cwd = os.getcwd()
            folder = 'Pickle files'
            CHECK_FOLDER = os.path.isdir(folder)
            newPath = os.path.join(cwd, folder)
            pickle_file_name = "{}.pickle".format(model)
            filename = '{}/{}'.format(newPath, pickle_file_name)
            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.mkdir(newPath)
            '''

            folder_name = 'Pickle files'
            pickle_file_name = "{}.pickle".format(model)

            filename = create_directory(pickle_file_name, folder_name)

            # dumping pickle file
            pickle.dump(model, open(filename, "wb"))
            logging.debug(f"""create_pickle_file is Successfully executed\n""")
        except Exception as e:
            logging.error(f""" ERROR IN : create_pickle_file : {str(e)}\n""")
            return f"""ERROR IN : create_pickle_file : {str(e)}\n"""

    @moniter
    def compare_accuracy(self, adj_r2_lin_model, adj_r2_lasso_model, adj_r2_ridge_model, adj_r2_elastic_net_model):
        try:
            models = pd.DataFrame({
                'Model': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet'],

                'Score': [adj_r2_lin_model, adj_r2_ridge_model, adj_r2_lasso_model, adj_r2_elastic_net_model]
            })

            folder_name = 'Accuracy Score'
            filename = 'accuracy_score.csv'
            file_name = create_directory(filename, folder_name)
            models.sort_values(by='Score', ascending=False).to_csv(file_name)
            logging.debug(f"""compare_accuracy is Successfully executed\n""")
            return models.sort_values(by='Score', ascending=False)
        except Exception as e:
            logging.error(f""" ERROR IN : compare_accuracy : {str(e)}\n""")
            return f"""ERROR IN : compare_accuracy : {str(e)}\n"""

    @moniter
    def predict_test(self, x_test, y_test, model):
        try:
            y_test_actual = y_test
            predictions = model.predict(x_test)

            # set the output as a dataframe and convert to csv file named submission.csv
            output = pd.DataFrame({'y_test_actual': y_test_actual, 'predictions': predictions})
            folder_name = 'Prediction file'
            filename = 'submission.csv'
            file_name = create_directory(filename, folder_name)
            output.to_csv(file_name, index=False)
            logging.debug(f"""predict_test is Successfully executed\n""")
            return output

        except Exception as e:
            logging.error(f""" ERROR IN : predict_test : {str(e)}\n""")
            return f"""ERROR IN : predict_test : {str(e)}\n"""

















