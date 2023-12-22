# It will contain that is used for reading data
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


# where to save the train,test,raw data,etc.
@dataclass  # provides all input things required for data ingestion component.
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    # artifact wil show us the output and will be stored there with filename train.csv
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # after running this class, the above 3 data paths will be saved inside ingestion_config

    def initiate_data_ingestion(self):
        # if we want to read data from database then it will be done inside this function
        logging.info("Entered the data ingestion method or component")
        # to check if any error occcurs, then check it with try and catch block
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")
            # create directories if doesn't exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of Data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    # Combining data injection and data transformation in below code
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    # this will print the r2 score
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
