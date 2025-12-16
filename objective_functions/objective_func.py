from demand_data import formatted_demand_df

# TODO : write correct code for this func
def objective_function(hourly_energy:list[float],
                       field_htf_pump_power:list[float],
                       pc_htf_pump_power:list[float])->float:
    """
    Calculates the objective function for optimisation
    
    :param hourly_energy: hourly energy
    :type hourly_energy: list[float]
    :param field_htf_pump_power: hourly field htf pumping power
    :type field_htf_pump_power: list[float]
    :param pc_htf_pump_power: hourly power cycle htf pumping power
    :type pc_htf_pump_power: list[float]
    """
    obj = hourly_energy * formatted_demand_df['Hourly Demand Met (in MW)'] - field_htf_pump_power*formatted_demand_df['Hourly Demand Met (in MW)'] - pc_htf_pump_power*formatted_demand_df['Hourly Demand Met (in MW)']

    return obj