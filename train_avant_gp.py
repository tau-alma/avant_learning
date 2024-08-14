import json
import avant_modeling.config as config
import torch
import numpy as np
import os
from datetime import datetime
from avant_modeling.utils import filter_data_batch, farthest_point_sampling, plot_sample_data, plot_error_density
from avant_modeling.gp import GPModel
from write_rosbag import make_result_rosbag


DATA_SOURCE_DIR = 'avant_identification_data/'
current_time = datetime.now().strftime("%Y_%m_%d-%H_%M")
RESULTS_DIR = f'avant_identification_results/{current_time}/'
os.makedirs(RESULTS_DIR, exist_ok=True)

data_idx_deriv_delay = [
    (1, False, 0, "beta"),                    # 0
    (2, False, 0, "omega"),                   # 1
    (3, False, 0, "dot_beta"),                # 2  
    (4, False, 0, "v_f"),                     # 3  
    (4, True, 0, "a_f"),                      # 4
    (3, True, 0, "dot_dot_beta"),             # 5  
    (5, False, 0, "steer"),                   # 6 
    (6, False, 0, "gas"),                     # 7 
    
    (1, False, 3, "truncated_beta_1"),        # 8
    (3, False, 3, "truncated_dot_beta_1"),    # 9
    (4, False, 3, "truncated_v_f_1"),         # 10
    (6, False, -3, "delayed_gas_1"),          # 11

    (1, False, 3, "truncated_beta_3"),        # 12
    (3, False, 3, "truncated_dot_beta_3"),    # 13
    (4, False, 3, "truncated_v_f_3"),         # 14
    (5, False, -3, "delayed_steer_3"),        # 15

    (3, True, 3, "truncated_dot_dot_beta_3"), # 16 
    (5, True, -3, "delayed_dot_steer_3"),     # 17 

    (4, True, 3, "truncated_a_f_1"),          # 18
    (6, True, -3, "delayed_dot_gas_1")        # 19

]

def compute_nominal_values(input_values, name):
    if "omega_f" in name:
        nominal = -(config.avant_lr * input_values[:, 2] + input_values[:, 3] * torch.sin(input_values[:, 0])) / (config.avant_lf * torch.cos(input_values[:, 0]) + config.avant_lr)
    elif "v_f" in name:
        nominal = 3*input_values[:, 11]
    elif any(val in name for val in ["dot_beta", "dot_dot_beta", "u_steer"]):
        a = 0.127 # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        b = 0.495 # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
        eps = eps0 - input_values[:, 12]
        # control signal gain (k) = the equation (6) page (3) in the paper
        k = 10 * a * b * torch.sin(eps) / torch.sqrt(a**2 + b**2 - 2*a*b*torch.cos(eps))
        if "dot_dot_beta" in name:
            nominal = (input_values[:, 17] * k) / k**2
        elif "dot_beta" in name:
            nominal = input_values[:, 15] / k
        elif "u_steer" in name:
            nominal = k * input_values[:, 13]
    elif "u_gas" in name:
        nominal = input_values[:, 10] / 3
    elif "a_f" in name:
        nominal = 3*input_values[:, 19]
    return nominal

if __name__ == '__main__':
    original_data, filtered_data = [], []
    per_file_eval_datas = []

    for file in os.listdir(DATA_SOURCE_DIR):
        if ".csv" not in file:
            continue
        print(f"Processing file: {file}")
        data_batch = np.loadtxt(os.path.join(DATA_SOURCE_DIR, file), skiprows=1, delimiter=",")
        timestamps, original, filtered = filter_data_batch(data_batch, data_idx_deriv_delay, os.path.join(DATA_SOURCE_DIR, file[:-4]))
        original_data.extend(original)
        filtered_data.extend(filtered)
        for i in range(len(timestamps)):
            print(timestamps[i].shape, original[:, i].shape)
        per_file_eval_datas.append((file, timestamps, original, filtered))

    original_data = np.asarray(original_data)
    filtered_data = np.asarray(filtered_data)
    print(f"Num data: {len(filtered_data)}")

    gp_datas = [
        (["beta", "dot beta", "v_f"], "omega_f", [0, 2, 3], 1),
        (["truncated_beta", "truncated_dot_beta", "delayed_u_gas"], "truncated_v_f", [8, 9, 11], 10),
        (["truncated_beta", "truncated_v_f", "delayed_u_steer"], "truncated_dot_beta", [12, 14, 15], 13),
        (["truncated_beta", "truncated_dot_beta", "delayed_u_steer", "delayed_dot_steer"], "truncated_dot_dot_beta", [12, 13, 15, 17], 16),

        (["truncated_beta", "truncated_dot_beta", "truncated_v_f"], "delayed_u_steer", [12, 13, 14], 15), 
        (["truncated_beta", "truncated_dot_beta", "truncated_v_f"], "delayed_u_gas", [8, 9, 10], 11),

        (["truncated_beta", "truncated_v_f", "delayed_u_gas", "delayed_dot_u_gas"], "truncated_a_f", [8, 9, 11, 19], 18)
    ]

    # For recreating a rosbag with the model outputs:
    always_include = [
        ("beta", 0),
        ("omega", 1),
        ("dot_beta", 2),
        ("v_f", 3),
        ("a_f", 4),
        ("dot_dot_beta", 5),
        ("steer", 6),
        ("gas", 7)
    ]
    result_bag_datas = {f: {
        "source_data": {},
        "original_inputs": {},
        "processed_inputs": {},
        "nominal_outputs": {},
        "corrected_outputs": {}
    } for f,_,_,_ in per_file_eval_datas}

    # Train the GPs independently:
    for input_names, output_name, input_indices, output_i in gp_datas:
        NAME_RESULT_DIR = os.path.join(RESULTS_DIR, output_name)
        os.makedirs(NAME_RESULT_DIR, exist_ok=True)

        # Train GP
        nominal = compute_nominal_values(torch.from_numpy(filtered_data), output_name)
        gp_inputs = torch.from_numpy(filtered_data[:, input_indices]).to(torch.float).cuda()
        gp_targets = torch.from_numpy(filtered_data[:, output_i]).to(torch.float).cuda()
        gp_targets -= nominal[:].to(torch.float).cuda()
        gp_model = GPModel(gp_inputs, gp_targets, train_epochs=200, train_lr=1e-1)
        torch.save(gp_inputs.cpu(), os.path.join(NAME_RESULT_DIR, f"{output_name}_gp_inputs.pth"))
        torch.save(gp_targets.cpu(), os.path.join(NAME_RESULT_DIR, f"{output_name}_gp_targets.pth"))
        torch.save(gp_model.cpu().state_dict(), os.path.join(NAME_RESULT_DIR, f"{output_name}_gp_model.pth"))
        with open(os.path.join(NAME_RESULT_DIR, f"{output_name}_gp_params.json"), "w") as outfile:
            outfile.write(json.dumps(gp_model.trained_params, indent=4))
        gp_model = gp_model.cuda()

        # Select GP dataset for evaluation by farthest point sampling:
        normalized_data = (filtered_data - np.mean(filtered_data, axis=0)) / (np.std(filtered_data, axis=0))
        # normalized_data = (filtered_data - np.min(filtered_data, axis=0)) / (np.max(filtered_data, axis=0) - np.min(filtered_data, axis=0))
        selected_idx = farthest_point_sampling(normalized_data[:, input_indices], num_samples=int(len(filtered_data)))
        plot_sample_data(filtered_data, selected_idx, os.path.join(NAME_RESULT_DIR, f"{output_name}_gp_samples.png"))

        # For error density plot:
        nominal_errors = []
        gp_errors = []
            
        # Evaluate the trained model for each data file:
        for file, timestamps, original_data, processed_data in per_file_eval_datas:
            FILE_RESULT_DIR = os.path.join(RESULTS_DIR, file)
            os.makedirs(FILE_RESULT_DIR, exist_ok=True)

            # Compute the model outputs using a GP fantasy model with a reduced number data points (since it may not be realistic to deploy with full dataset)
            gp_eval_inputs = torch.from_numpy(original_data[:, input_indices]).to(torch.float).cuda()
            with torch.no_grad():
                gp_model.fantasy_model(gp_inputs[selected_idx], gp_targets[selected_idx])
                gp_outputs = gp_model(gp_eval_inputs)
            mean = gp_outputs.mean.cpu().numpy()
            nominal = compute_nominal_values(torch.from_numpy(original_data), output_name).numpy()
            mean += nominal

            # Data from the source rosbag:
            for name, idx in always_include:
                result_bag_datas[file]["source_data"][name] = (timestamps[idx], original_data[:, idx])

            # new GP model input values (e.g. delayed signals)
            for name_idx, input_idx in enumerate(input_indices):
                input_name = f"{output_name}_gp__{input_names[name_idx]}_in"
                result_bag_datas[file]["original_inputs"][input_name] = (timestamps[input_idx], original_data[:, input_idx])
                result_bag_datas[file]["processed_inputs"][input_name] = (timestamps[input_idx], processed_data[:, input_idx])

            result_bag_datas[file]["nominal_outputs"][output_name] = (timestamps[output_i], nominal)
            result_bag_datas[file]["corrected_outputs"][output_name] = (timestamps[output_i], mean)

            scaler = 1
            if any (n in output_name for n in ["omega_f", "dot_beta", "dot_dot_beta"]):
                scaler = 180/np.pi
            nominal_error = np.abs(scaler*original_data[:, output_i] - scaler*nominal)
            gp_error = np.abs(scaler*original_data[:, output_i] - scaler*mean)
            nominal_errors.extend(nominal_error)
            gp_errors.extend(gp_error)

        nominal_errors = np.asarray(nominal_errors)
        gp_errors = np.asarray(gp_errors)
        plot_error_density(np.c_[nominal_errors, gp_errors], name=output_name, out_file=f"{RESULTS_DIR}/{output_name}_error_densities.png")

    make_result_rosbag(f"{RESULTS_DIR}/rosbags", result_bag_datas)