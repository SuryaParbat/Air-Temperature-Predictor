import data_cleaner
from data_cleaner import DataCleaner
import lr_models, logging
from lr_models import LinearRegressionModels
from create_new_directory import create_directory

class Prediction:

    def predict(self):
        """
        In this function we are going to call all the function from data_cleaner.py and lr_models.py
        :return:
        """

        try:
            data_cleaner = DataCleaner()

            # Calling load_data to create dataframe of given csv file
            data_frame = data_cleaner.load_data("Data//ai4i2020.csv")
            print(data_frame)

            # Calling pandas_profiling to get profile report of the dataframe
            data_cleaner.pandas_profiling(data_frame)

            # Calling prepare_data to split data into features ie. x and target variable ie. y
            x,y = data_cleaner.prepare_data(data_frame)
            print(x)
            print(y)

            # Calling feature_standard to convert all the features into same level
            std_arr = data_cleaner.feature_standard(x)
            print(std_arr)

            # Calling vif_socre to get VIF Score
            vif_score = data_cleaner.vif_socre(x, std_arr)
            print(vif_score)

            # Calling spliting_train_test to split data into training and testing.
            x_train, x_test, y_train, y_test =data_cleaner.spliting_train_test(std_arr,y)
            print(x_train)
            print(y_train)

            lr_models = LinearRegressionModels()

            # Calling lr_model to get accuracy of linear regression model
            lin_reg_accuracy, lin_reg = lr_models.lr_model(x_train, x_test, y_train, y_test)
            print(lin_reg_accuracy)

            # Calling lr_model to get accuracy of linear regression model
            lasso_model_accuracy, lasso = lr_models.lasso_model(x_train, x_test, y_train, y_test)
            print(lasso_model_accuracy)

            # Calling lr_model to get accuracy of linear regression model
            ridge_model_accuracy, ridge = lr_models.ridge_model(x_train, x_test, y_train, y_test)
            print(ridge_model_accuracy)

            # Calling lr_model to get accuracy of linear regression model
            elactic_model_accuracy, elastic_lr = lr_models.elastic_net_model(x_train, x_test, y_train, y_test)
            print(elactic_model_accuracy)

            # Calling adj_r2 to get Adjusted R2 score of all models
            adj_r2_lin_model = lr_models.adj_r2(x_test,y_test, lin_reg)
            adj_r2_lasso_model = lr_models.adj_r2(x_test,y_test, lasso)
            adj_r2_ridge_model = lr_models.adj_r2(x_test,y_test, ridge)
            adj_r2_elastic_net_model = lr_models.adj_r2(x_test,y_test, elastic_lr)
            # print(adj_r2_lin_model)
            # print(adj_r2_lasso_model)
            # print(adj_r2_ridge_model)
            # print(adj_r2_elastic_net_model)

            # Creating Pickle file of all models
            models = [lin_reg, lasso, ridge, elastic_lr]
            for model in models:
                lr_models.create_pickle_file(model)

            # Comparing all the accuracy scores
            accuracy_score = lr_models.compare_accuracy(adj_r2_lin_model, adj_r2_lasso_model, adj_r2_ridge_model, adj_r2_elastic_net_model)
            print(accuracy_score)

            # Testing model with x_test
            output = lr_models.predict_test(x_test, y_test, lasso)
            print(output)

            logging.debug(f"""predict is Successfully executed\n""")

        except Exception as e:
            logging.error(f""" ERROR IN : adj_r2 : {str(e)}\n""")
            return f"""ERROR IN : adj_r2 : {str(e)}\n"""


