import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

            mlflow.set_registry_uri("https://dagshub.com/mkumawat1307/Project_Diamond_Price_Prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = model.predict(x_test)
                (rmse, mae,r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # this condition is for dagshub
                # model registry does not work with file store
                if tracking_url_type_store != "file":
                    

                    # register the model
                    # there are other ways to use the model registry, which depends on the yse case,
                    # please refer to the doc for more information
                    # https://mlflow.org/docs/latest/model-registry.html#ip-workflow
                    mlflow.sklearn.load_model(model, "model", registered_model_name="ml_model")
                # it is for the local
                else:
                    mlflow.sklearn.log_model(model,"model")
                


        except Exception as e:
            raise e
