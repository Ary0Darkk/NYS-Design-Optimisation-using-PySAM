import json
import pprint
import PySAM.TroughPhysical as TP
from config import CONFIG
from utilities.list_nesting import replace_1st_order


def run_simulation(overrides: None = None):
    # overrides = dict(overrides)

    tp = TP.default(CONFIG["model"])

    # Load JSON
    with open(CONFIG["json_file"], "r") as f:
        data = json.load(f)

    # assign(dict) -> None : takes dict and copies the values into the PySAM model
    # export() -> dict : pulls every single parameter currently set in that
    # PySAM module and puts them into a standard Python dictionary
    # replace(dict) -> None : If your dictionary only has 5 variables,
    # PySAM will "unassign" or clear all other variables in that module that are not in your dictionary.
    # NOTE : can't use assign here, because we are assigning
    # variables of different sub-modules
    for k, v in data.items():
        if k != "number_inputs":
            try:
                tp.value(k, v)
            except Exception:
                pass
    # Apply overrides (changed parameters)
    if overrides is not None:
        for k, v in overrides.items():
            # print(f'Value before : {tp.value(k)}')
            # Fetch current value once (Best practice for performance)
            current_val = tp.value(k)

            # Case: Scalar (int, float, or boolean)
            if isinstance(current_val, (int, float, bool)):
                tp.value(k, v)

            #  Case: 1st Order Sequence (List or Tuple)
            elif isinstance(current_val, (list, tuple)):
                if len(current_val) > 0:
                    # Use your 1st order function to swap the first element
                    v_new = replace_1st_order(data=current_val, new_val=v)
                    tp.value(k, v_new)
                else:
                    # If the list is empty, we initialize it with the new value
                    tp.value(k, [v] if isinstance(current_val, list) else (v,))

            # 4. Optional: Log or skip if it's an unexpected type (like a string)
            else:
                print(f"Skipping key {k}: Unknown type {type(current_val)}")
            # print(f'Value after : {tp.value(k)}')
    tp.execute()

    # sim_result = {
    #     "hourly_energy": tp.Outputs.P_out_net,
    #     "pc_htf_pump_power": tp.Outputs.cycle_htf_pump_power,
    #     "field_htf_pump_power": tp.Outputs.W_dot_field_pump,
    #     "Field collector optical end loss":tp.Outputs.EndLoss_ave,
    #     "Parasitic power fixed load":tp.Outputs.P_fixed,
    #     "gross [MWe]":tp.Outputs.P_cycle,
    #     "Parasitic power generation-dependent load":tp.Outputs.P_plant_balance_tot,
    #     "Field collector row shadowing loss":tp.Outputs.RowShadow_ave,
    #     "Field collector tracking power [MWe]":tp.Outputs.W_dot_sca_track,

    # }

    sim_result = tp.export()

    return sim_result  # dict of outputs


if __name__ == "__main__":
    # override= {

    # }

    result = run_simulation()

    # with open("experiments/sim_result.json","w")as f:
    #     json.dump(result,f)

    pprint.pprint(result)
