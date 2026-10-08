# import socket
# import pickle
# import struct
# import multiprocessing as mp

# from config import CONFIG
# from ...simulation.simulation import run_simulation
# from ...objective_functions.objective_func import objective_function


# HOST = "0.0.0.0"
# PORT = 5000


# def run_one_simulation(individual_id, day_index, individual):
#     design_params = individual[:5]
#     operational_params = individual[5:]

#     start = day_index * 3
#     end = start + 3

#     daily_operational = operational_params[start:end]

#     simulation_input = design_params + daily_operational

#     # Run PySAM
#     sim_result, penalty_flag = run_simulation(simulation_input)

#     if penalty_flag:
#         return (
#             individual_id,
#             day_index,
#             CONFIG["penalty"],
#         )

#     var_names = CONFIG["design"]["overrides"] + CONFIG["operational"]["overrides"]

#     var_types = CONFIG["design"]["types"] + CONFIG["operational"]["types"]

#     overrides = {
#         var_names[j]: var_types[j](simulation_input[j]) for j in range(len(var_names))
#     }

#     objective = objective_function(
#         sim_result["hourly_energy"],
#         sim_result["pc_htf_pump_power"],
#         sim_result["field_htf_pump_power"],
#         sim_result["field_collector_tracking_power"],
#         sim_result["pc_startup_thermal_power"],
#         sim_result["field_piping_thermal_loss"],
#         sim_result["receiver_thermal_loss"],
#         f_overrides=overrides,
#         day_index=day_index,
#     )

#     if objective is None:
#         objective = CONFIG["penalty"]

#     return (
#         individual_id,
#         day_index,
#         float(objective),
#     )


# def evaluate_batch(batch, pool):
#     tasks = []

#     for individual_id, individual in enumerate(batch):
#         for day_index in range(CONFIG["num_days"]):
#             tasks.append(
#                 (
#                     individual_id,
#                     day_index,
#                     individual,
#                 )
#             )

#     print(f"Worker created {len(tasks)} simulation tasks")

#     results = pool.starmap(
#         run_one_simulation,
#         tasks,
#     )

#     fitnesses = [0.0 for _ in batch]

#     for individual_id, day_index, objective in results:
#         fitnesses[individual_id] += objective

#     return [(fitness,) for fitness in fitnesses]


# def recv_data(conn):
#     raw_length = conn.recv(4)

#     if not raw_length:
#         return None

#     message_length = struct.unpack("!I", raw_length)[0]

#     data = b""

#     while len(data) < message_length:
#         packet = conn.recv(min(65536, message_length - len(data)))

#         if not packet:
#             raise ConnectionError("Connection closed")

#         data += packet

#     return pickle.loads(data)


# def send_data(conn, data):
#     payload = pickle.dumps(data)

#     conn.sendall(struct.pack("!I", len(payload)) + payload)


# def main():
#     print("Starting worker...")

#     pool = mp.Pool(processes=112)

#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

#     server.bind((HOST, PORT))

#     server.listen(1)

#     print(f"Worker listening on port {PORT}")

#     conn, addr = server.accept()

#     print(f"Master connected from {addr}")

#     try:
#         while True:
#             batch = recv_data(conn)

#             if batch is None:
#                 break

#             print(f"Received batch with {len(batch)} individuals")

#             fitnesses = evaluate_batch(batch, pool)

#             send_data(conn, fitnesses)

#     finally:
#         conn.close()
#         server.close()

#         pool.close()
#         pool.join()

#         print("Worker stopped")


# if __name__ == "__main__":
#     main()
