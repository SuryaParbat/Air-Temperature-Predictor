import pandas as pd
import logging, os, pickle
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from create_new_directory import create_directory

from logger import create_log_file, moniter

create_log_file('TempPrediction.log')


class DataCleaner:

    @moniter
    def load_data(self, data):
        """
        Inn this function we are loading csv file and converting it into
        DataFrame using pandas.
        :param data:
        :return:
        """

        try:
            df = pd.read_csv(data)
            logging.debug(f""" Data file : {data} is Successfully loaded\n""")
            # return f""" Data file : {data} is Successfully loaded\n"""
            return df
        except Exception as e:
            logging.error(f""" ERROR IN : load_data : {str(e)}\n""")
            return f"""ERROR IN : load_data : {str(e)}\n"""

    @moniter
    def pandas_profiling(self, df):
        """
        In this function we doing profiling of data and generating profiling report
        by using ProfileReport.
        :param df:
        :return:
        """
        try:

            folder_name = 'Pandas Profiling Report'
            profile_report_name = "Pandas_Profile_Report.html"

            filename = create_directory(profile_report_name, folder_name)
            profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            profile.to_file(filename)
            logging.debug(f"""pandas_profiling is Successfully loaded\n""")
        except Exception as e:
            logging.error(f""" ERROR IN : pandas_profiling : {str(e)}\n""")
            return f"""ERROR IN : pandas_profiling : {str(e)}\n"""

    """
    From the pandas profiling report we can see that there is no missing values in
    the dataset, so there is no need for handling missing values.

    However, we need to standardize our dataset into same level. For that we are going to use
    StandardScaler().

    And we need to check mutic0lineartity between all features by using VIF score(>10, reject).
    """
    @moniter
    def handling_missing_values(self):
        pass

    """
    Now dividing dataframe into features i.e. x and target variable i.e. y.
    """
    @moniter
    def prepare_data(self, df):
        """
        In this function we are going to make feature variable and target variable.
        feature variable x contains all the required feature columns.
        target variable contains values which we want to predict i.e y
        :param df:
        :return:
        """

        try:
            x = df.drop(columns=["Air temperature [K]", "UDI", "Product ID", "Type", "Machine failure"])
            y = df['Air temperature [K]']
            logging.debug(f"""prepare_data is Successfully loaded and DataFrame is
            Successfully divided into features and target variable\n""")
            return x,y

        except Exception as e:
            logging.error(f""" ERROR IN : prepare_data : {str(e)}\n""")
            return f"""ERROR IN : prepare_data : {str(e)}\n"""

    """
    Now we need to standardize our features into same level by using
    StandardScaler()
    """

    @moniter
    def feature_standard(self, x):
        """
        In this function we are trying to get all the features onto the same level
        by using StandardScaler()
        And we are also saving StandardScaler into a pickle file for using it in prediction.
        :param x:
        :return:
        """
        try:

            folder_name = 'Pickle files'
            pickle_file_name = "standard_scaler.pickle"

            filename = create_directory(pickle_file_name, folder_name)

            scaler = StandardScaler()
            arr = scaler.fit_transform(x)
            pickle.dump(scaler, open(filename, "wb"))
            logging.debug(f"""feature_standard is Successfully loaded\n""")
            logging.debug(f"""features are Successfully standardize by StandardScaler() \n""")
            logging.debug(f"""StandardScaler() is Successfully dumped into pickle file\n""")
            return arr
        except Exception as e:
            logging.error(f""" ERROR IN : feature_standard : {str(e)}\n""")
            return f"""ERROR IN : feature_standard : {str(e)}\n"""

    """
    Now we need to check VIF score so that we can know, if there is any correlation between
    features.
    if any features are correlated than we need to drop one of those features.
    """

    @moniter
    def vif_socre(self,x, arr):
        """
        In this function we are going to check if there is any correlation between feature
        columns by using VIF SCORE.
        :param x:
        :param arr:
        :return:
        """

        try:
            folder_name = 'VIF CSV'
            filename = "Final_vif.csv"
            file_name = create_directory(filename, folder_name)

            vif_df = pd.DataFrame()
            vif_df["vif"] = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
            vif_df['feature'] = x.columns
            vif_df.to_csv(file_name)

            logging.debug(f"""vif_socre is Successfully loaded\n""")
            return vif_df

        except Exception as e:
            logging.error(f""" ERROR IN : vif_socre : {str(e)}\n""")
            return f"""ERROR IN : vif_socre : {str(e)}\n"""

    """
    From VIF score we can clearly see that Machine failure feature has high vif score i.e >10
    so we need to drop that column.
    
    After getting final features and target variables we need to split them into
    training and testing datasets.
    """

    @moniter
    def spliting_train_test(self, arr,y):
        """
        After getting final features and target variables we need to split them into
        training and testing datasets by using train_test_split.

        :param arr:
        :param y:
        :return:
        """

        try:
            x_train, x_test, y_train, y_test = train_test_split(arr, y, test_size=0.2, random_state=100)
            logging.debug(f"""spliting_train_test is Successfully loaded\n""")
            logging.debug(f"""Successfully splited freatures and traget variable into train test\n""")
            return x_train, x_test, y_train, y_test

        except Exception as e:
            logging.error(f""" ERROR IN : spliting_train_test : {str(e)}\n""")
            return f"""ERROR IN : spliting_train_test : {str(e)}\n"""










