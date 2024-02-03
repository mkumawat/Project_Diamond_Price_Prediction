from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer
from src.DimondPricePrediction.components.model_evaluation import ModelEvaluation

import os
import sys
import pandas as pd 
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model_traner_obj=ModelTrainer()
model_traner_obj.initate_model_training(train_arr,test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr, test_arr)
