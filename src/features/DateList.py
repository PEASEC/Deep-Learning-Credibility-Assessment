from typing import Dict
from datetime import datetime


class HolidayIndex:
    holidays: Dict[datetime.date, str] = None

    @staticmethod
    def initialize():
        if HolidayIndex.holidays is None:
            HolidayIndex.holidays = dict()
            with open("resources/dates/holidays.csv", "r") as file:
                for index, row in enumerate(file):
                    if index > 0:
                        str_date, day, day_type = row.split(",")
                        date = datetime.strptime(str_date, "%Y-%m-%d").date()
                        HolidayIndex.holidays[date] = day

    @staticmethod
    def is_holiday(date: datetime.date) -> bool:
        return date in HolidayIndex.holidays
