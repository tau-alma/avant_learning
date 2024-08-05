import json
import avant_modeling.config as config
import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import interpolate
from avant_modeling.utils import filter_data_batch, farthest_point_sampling, plot_sample_data, plot_error_density
from avant_modeling.gp import GPModel

DATA_SOURCE_DIR = 'avant_identification_data/'
current_time = datetime.now().strftime("%H_%M_%S")
RESULTS_DIR = f'avant_identification_results/{current_time}/'
os.makedirs(RESULTS_DIR, exist_ok=True)

data_idx_deriv_delay = [
    (1, False, 0, "beta"),                  # 0
    (2, False, 0, "omega"),                 # 1
    (3, False, 0, "dot_beta"),              # 2  
    (4, False, 0, "v_f"),                   # 3  
    (5, False, 0, "steer"),                 # 4 
    (6, False, 0, "gas"),                   # 5 
    
    (6, False, -1, "delayed_gas_1"),        # 6
    (1, False, 1, "truncated_beta_1"),      # 7
    (4, False, 1, "truncated_v_f_1"),       # 8

    (1, False, 3, "truncated_beta_3"),        # 9
    (4, False, 3, "truncated_v_f_3"),         # 10
    (5, False, -3, "delayed_steer_3"),        # 11
    (3, False, 3, "truncated_dot_beta_3"),    # 12

    (3, True, 3, "truncated_dot_dot_beta_3"), # 13 
    (5, True, -3, "delayed_dot_steer_3"),     # 14 
]

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
        per_file_eval_datas.append((file, timestamps, original))

    original_data = np.asarray(original_data)
    filtered_data = np.asarray(filtered_data)

    print(f"Num data: {len(filtered_data)}")

    # Train the GPs independently:
    gp_datas = [
        ("omega_f", [0, 2, 3], 1),               # inputs: beta, dot beta, v_f
        ("v_f", [7, 6], 8),                      # inputs: beta and gas
        ("dot_beta", [9, 10, 11], 12),           # inputs: truncated_beta, truncated_v_f and delayed_steer
        ("dot_dot_beta", [9, 12, 11, 14], 13),   # inputs: truncated_beta, truncated_dot_beta, delayed_steer and delayed_dot_steer

        ("u_steer", [9, 12, 10], 11),            # inputs: truncated_beta, truncated_dot_beta and truncated_v_f 
        ("u_gas", [7, 8], 8)                     # inputs: truncated_beta and truncated_v_f
    ]
    for name, input_indices, output_i in gp_datas:
        NAME_RESULT_DIR = os.path.join(RESULTS_DIR, name)
        os.makedirs(NAME_RESULT_DIR, exist_ok=True)

        if name == "omega_f":
            nominal = -(config.avant_lr * filtered_data[:, 2] + filtered_data[:, 3] * np.sin(filtered_data[:, 0])) / (config.avant_lf * np.cos(filtered_data[:, 0]) + config.avant_lr)
        elif name == "v_f":
            nominal = 3*filtered_data[:, 6]
        elif name in ["dot_beta", "dot_dot_beta", "u_steer"]:
            a = 0.127 # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
            b = 0.495 # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
            eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
            eps = eps0 - filtered_data[:, 9]
            # control signal gain (k) = the equation (6) page (3) in the paper
            k = 10 * a * b * np.sin(eps) / np.sqrt(a**2 + b**2 - 2*a*b*np.cos(eps))
            if name == "dot_beta":
                nominal = filtered_data[:, 11] / k
            elif name == "dot_dot_beta":
                nominal = (filtered_data[:, 14] * k) / k**2
            elif name == "u_steer":
                nominal = k * filtered_data[:, 12]
        elif name == "u_gas":
            nominal = filtered_data[:, 8] / 3
        nominal = torch.from_numpy(nominal)

        gp_inputs = torch.from_numpy(filtered_data[:, input_indices]).to(torch.float).cuda()
        gp_targets = torch.from_numpy(filtered_data[:, output_i]).to(torch.float).cuda() - nominal[:].to(torch.float).cuda()
        gp_model = GPModel(gp_inputs, gp_targets, train_epochs=100, train_lr=1e-1)
        torch.save(gp_inputs.cpu(), os.path.join(NAME_RESULT_DIR, f"{name}_gp_inputs.pth"))
        torch.save(gp_targets.cpu(), os.path.join(NAME_RESULT_DIR, f"{name}_gp_targets.pth"))
        torch.save(gp_model.cpu().state_dict(), os.path.join(NAME_RESULT_DIR, f"{name}_gp_model.pth"))
        params = {k: v.cpu().numpy().tolist() for k, v in gp_model.state_dict().items()}
        with open(os.path.join(NAME_RESULT_DIR, f"{name}_gp_params.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4))
        gp_model = gp_model.cuda()

        scaler = 1
        if name in ["omega_f", "dot_beta", "dot_dot_beta"]:
            scaler = 180/np.pi

        # Select GP dataset for evaluation by farthest point sampling:
        normalized_data = (filtered_data - np.mean(filtered_data, axis=0)) / (np.std(filtered_data, axis=0))
        # normalized_data = (filtered_data - np.min(filtered_data, axis=0)) / (np.max(filtered_data, axis=0) - np.min(filtered_data, axis=0))
        selected_idx = farthest_point_sampling(normalized_data[:, input_indices], num_samples=int(1000))
        plot_sample_data(filtered_data, selected_idx, os.path.join(NAME_RESULT_DIR, f"{name}_samples.png"))

        nominal_errors = []
        gp_errors = []
        # Evaluate the trained model for each data file:
        for file, timestamps, data in per_file_eval_datas:
            FILE_RESULT_DIR = os.path.join(RESULTS_DIR, file)
            os.makedirs(FILE_RESULT_DIR, exist_ok=True)

            gp_eval_inputs = torch.from_numpy(data[:, input_indices]).to(torch.float).cuda()
            with torch.no_grad():
                gp_model.fantasy_model(gp_inputs[selected_idx], gp_targets[selected_idx])
                gp_outputs = gp_model(gp_eval_inputs)
            mean = gp_outputs.mean.cpu().numpy()[:, 0]

            if name == "omega_f":
                nominal = -(config.avant_lr * data[:, 2] + data[:, 3] * np.sin(data[:, 0])) / (config.avant_lf * np.cos(data[:, 0]) + config.avant_lr)
            elif name == "v_f":
                nominal = 3*data[:, 6]
            elif name in ["dot_beta", "dot_dot_beta", "u_steer"]:
                a = 0.127 # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
                b = 0.495 # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
                eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
                eps = eps0 - data[:, 9]
                # control signal gain (k) = the equation (6) page (3) in the paper
                k = 10 * a * b * np.sin(eps) / np.sqrt(a**2 + b**2 - 2*a*b*np.cos(eps))
                if name == "dot_beta":
                    nominal = data[:, 11] / k
                elif name == "dot_dot_beta":
                    nominal = (data[:, 14] * k) / k**2
                elif name == "u_steer":
                    nominal = k * data[:, 12]
            elif name == "u_gas":
                nominal = data[:, 8] / 3
            mean += nominal

            nominal_error = np.abs(scaler*data[:, output_i] - scaler*nominal)
            gp_error = np.abs(scaler*data[:, output_i] - scaler*mean)
            nominal_errors.extend(nominal_error)
            gp_errors.extend(gp_error)

        nominal_errors = np.asarray(nominal_errors)
        gp_errors = np.asarray(gp_errors)
        plot_error_density(np.c_[nominal_errors, gp_errors], name=name, out_file=f"{RESULTS_DIR}/{name}_error_density.png")
