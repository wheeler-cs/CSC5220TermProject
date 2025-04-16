import NREL_API

import datetime
import os
import pandas as pd
import requests
from time import sleep
from tqdm import tqdm
from typing import List


# ======================================================================================================================
def load_data(data_dir: str = "./cleaned_data") -> List[pd.DataFrame]:
    '''
    
    '''
    # Adapted from Sam's code in data_loading/vehicle_dataset.py
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in tqdm(file_list, desc="Loading Data Files")]
    return dataframes


def create_estimate_frames(data: List[pd.DataFrame]) -> pd.DataFrame:
    '''
    
    '''
    new_dataframe = pd.DataFrame()
    for frame in data:
        start_time = gps_time_convert(frame["GPS Time"].head(1).iloc[0])
        end_time   = gps_time_convert(frame["GPS Time"].tail(1).iloc[0])
        duration   = end_time - start_time
        avg_speed  = frame["Speed (OBD)(mph)"].mean()
        avg_grade  = frame["Grade"].mean()
        new_data = pd.DataFrame([{"start": start_time,
                                  "end": end_time,
                                  "duration": duration,
                                  "calc_distance": (duration.total_seconds() / 3600) * avg_speed,
                                  "avg_speed": avg_speed,
                                  "avg_grade": avg_grade,
                                  "routee_estimate": 0.0}])
        new_dataframe = pd.concat([new_dataframe, new_data], ignore_index=True)
    return new_dataframe


def gps_time_convert(gps_time: str) -> datetime.datetime:
    '''
    
    '''
    parsed_datetime = gps_time.split(' ')
    parsed_time = parsed_datetime[3].split(':')
    converted_time = datetime.datetime(year=int(parsed_datetime[5]),
                                       month=datetime.datetime.strptime(parsed_datetime[1], "%b").month,
                                       day=int(parsed_datetime[2]),
                                       hour=int(parsed_time[0]),
                                       minute=int(parsed_time[1]),
                                       second=int(parsed_time[2]))
    return converted_time


def get_averages(data: List[pd.DataFrame], parameter: str) -> List[float]:
    '''

    '''
    avgs = [frame[parameter].mean().item() for frame in data]
    return avgs


def calculate_distance(avg_speeds: List[float], durations: List[datetime.timedelta]) -> List[float]:
    '''
    
    '''
    miles = []
    for i in range(0, len(avg_speeds)):
        miles.append(avg_speeds[i] * (durations[i].total_seconds() / 3600))
    return miles


def make_requests(req: NREL_API.Request, data: pd.DataFrame, time_delay: int = 5) -> pd.DataFrame:
    '''

    '''
    data["routee_estimate"] = None
    for i in tqdm(range(len(data)), desc="Contacting RouteE"):
        sleep(time_delay)
        row = data.iloc[i]
        req.miles         = row["calc_distance"]
        req.speed_mph     = row["avg_speed"]
        req.grade_percent = row["avg_grade"]
        data.loc[i, "routee_estimate"] = req.make_request().json()["route"][0]["energy_estimate"]
    return data    




# ======================================================================================================================
if __name__ == "__main__":
    dataframes = load_data()
    estimates = create_estimate_frames(dataframes)
    req = NREL_API.Request(model="2016_TOYOTA_Corolla_4cyl_2WD", id=1, miles=0, speed_mph=0, grade_percent=0)
    req.load_api_key("./RouteE.key")
    estimates = make_requests(req, estimates, time_delay=3)
    with open("estimates.json", 'w') as results:
        estimates.to_json(results, orient="records", lines=True, indent=4)
