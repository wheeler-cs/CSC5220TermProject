import argparse
from dataclasses import dataclass
import requests

DEFAULT_URL = "https://developer.nrel.gov/api/routee/v3/powertrain/route"
DEFAULT_KEY = {"api_key": "DEMO_KEY"}


@dataclass
class Request:
    car_model: str
    id: int
    miles: int
    speed_mph: int
    grade_percent: int


def make_request(req: Request) -> requests.Response:
    data = {"powertrain_model": req.car_model,
            "links":
                [{"id": req.id,
                  "miles": req.miles,
                  "speed_mph": req.speed_mph,
                  "grade_percent": req.grade_percent}]
            }
    return(requests.post(DEFAULT_URL, params=DEFAULT_KEY, json=data))


if __name__ == "__main__":
    vehicle_request = Request(car_model="2016_TOYOTA_Corolla_4cyl_2WD", id=1, miles=20, speed_mph=40, grade_percent=10)
    response = make_request(vehicle_request)
    print(response.text)
