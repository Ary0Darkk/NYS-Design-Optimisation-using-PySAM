# hour sampling file
# from config import CONFIG
from datetime import date


def build_operating_hours_from_month_day(
    user_days: dict[str, list[tuple[int, int]]],
    start_hour: int = 7,
    end_hour: int = 16,
    year: int = 2020,
):
    """
    Returns a list of records with both:
    - human-readable time info
    - SAM-compatible hour index
    """

    records = []

    for season, dates in user_days.items():
        for month, day in dates:
            doy = date(year, month, day).timetuple().tm_yday
            day_start = (doy - 1) * 24  # SAM: hour 1 = Jan 1, 00–01

            for hour_of_day in range(start_hour, end_hour + 1):
                sam_hour = day_start + hour_of_day + 1

                records.append(
                    {
                        "season": season,
                        "month": month,
                        "day": day,
                        "hour_of_day": hour_of_day,
                        "sam_hour": sam_hour,
                    }
                )

    return records


# OPERATING_HOURS = build_operating_hours_from_days(CONFIG["USER_DEFINED_DAYS"])

# print(OPERATING_HOURS)
