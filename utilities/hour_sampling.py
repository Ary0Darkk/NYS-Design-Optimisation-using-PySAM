# hour sampling file
from datetime import date


def build_operating_days_from_month_day(
    user_days: dict[str, list[tuple[int, int]]],
    year: int = 2020,
):
    """
    Returns day-level records for routing pipeline.

    Each record contains:
    - season
    - month, day
    - day_of_year (1–365)
    - day_index (0–364)
    """

    records = []

    for season, dates in user_days.items():
        for month, day in dates:
            doy = date(year, month, day).timetuple().tm_yday  # 1-based
            day_index = doy - 1  # 0-based index

            records.append(
                {
                    "season": season,
                    "month": month,
                    "day": day,
                    "day_of_year": doy,
                    "day_index": day_index,
                }
            )

    return records
