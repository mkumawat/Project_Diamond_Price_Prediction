import os
import sys
from sklearn.matrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np 
import pickle
from src.DimondPricePrediction.utils.utils import load_object

class ModelEvaluation:
    def __init_(self):
        pass

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae= mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse,mae,r2
    
    def initiate_model_evaluation(self,train_array,test_array):
        try:
            x_test, y_test = (test_array[:,:-1], test_array[:,-1])

            model_path = os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = model.predict(x_test)
                (rmse, mae,r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # model registry does not work with file store
                if tracking_url_type_store != "file":
                    pass

                    # register the model
                    # there are other ways to use the model 


        except Exception as e:
            pass
