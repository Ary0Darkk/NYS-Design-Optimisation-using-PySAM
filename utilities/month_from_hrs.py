# func to convert hourly data into monthly
def hrs_to_months(input_data: list) -> list:
    input_data = list(input_data)
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    idx = 0
    output_data = []
    for len_month in days_in_months:
        # define index
        stop_idx = idx + len_month * 24

        # sum energy for month
        sum_of_energy = sum(input_data[idx:stop_idx])

        # add to list
        output_data.append(sum_of_energy)

        # index update
        idx = stop_idx

    return output_data
