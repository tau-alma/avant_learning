import json
import avant_modeling.config as config
import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import interpolate
from avant_modeling.utils import filter_data_batch, farthest_point_sampling, plot_sample_data
from avant_modeling.gp import GPModel

DATA_SOURCE_DIR = 'avant_identification_data/'
current_time = datetime.now().strftime("%H_%M_%S")
RESULTS_DIR = f'avant_identification_results/{current_time}/'
os.makedirs(RESULTS_DIR, exist_ok=True)

data_idx_deriv_delay = [
    (1, False, 0, "x"),                     # 0
    (2, False, 0, "y"),                     # 1
    (3, False, 0, "theta"),                 # 2
    (4, False, 0, "beta"),                  # 3
    (5, False, 0, "omega"),                 # 4
    (6, False, 0, "dot_beta"),              # 5  
    (7, False, 0, "v_f"),                   # 6  
    (8, False, 0, "steer"),                 # 7 
    (9, False, 0, "gas"),                   # 8 

    (4, False, 3, "truncated_beta"),        # 9
    (7, False, 3, "truncated_v_f"),         # 10
    (8, False, -3, "delayed_steer"),        # 11
    (6, False, 3, "truncated_dot_beta"),    # 12

    (6, True, 3, "truncated_dot_dot_beta"), # 13 
    (8, True, -3, "delayed_dot_steer"),     # 14 
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
        per_file_eval_datas.append((file, timestamps, filtered))

    original_data = np.asarray(original_data)
    filtered_data = np.asarray(filtered_data)

    print(f"Num data: {len(filtered_data)}")

    # Train the GPs independently:
    gp_datas = [
        ("omega_f", [3, 5, 6, 7], 4),            # inputs: beta, dot beta, v_f and steer
        ("v_f", [3, 8], 6),                      # inputs: beta and gas
        ("dot_beta", [9, 10, 11], 12),           # inputs: truncated_beta, truncated_v_f and delayed_steer
        ("dot_dot_beta", [9, 12, 11, 14], 13)    # inputs: truncated_beta, truncated_dot_beta, delayed_steer and delayed_dot_steer
    ]
    trained_gps = []
    for name, input_indices, output_i in gp_datas:
        NAME_RESULT_DIR = os.path.join(RESULTS_DIR, name)
        os.makedirs(NAME_RESULT_DIR, exist_ok=True)

        if name == "omega_f":
            nominal = -(config.avant_lr * filtered_data[:, 5] + filtered_data[:, 6] * np.sin(filtered_data[:, 3])) / (config.avant_lf * np.cos(filtered_data[:, 3]) + config.avant_lr)
        elif name == "v_f":
            nominal = 3.5*filtered_data[:, 8]
        elif name in ["dot_beta", "dot_dot_beta"]:
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
        nominal = torch.from_numpy(nominal)

        gp_inputs = torch.from_numpy(filtered_data[:, input_indices]).to(torch.float).cuda()
        gp_targets = torch.from_numpy(filtered_data[:, output_i]).to(torch.float).cuda() - nominal[:].to(torch.float).cuda()
        gp_model = GPModel(gp_inputs, gp_targets, train_epochs=100, train_lr=1e-1)
        torch.save(gp_inputs, os.path.join(NAME_RESULT_DIR, f"{name}_gp_inputs.pth"))
        torch.save(gp_targets, os.path.join(NAME_RESULT_DIR, f"{name}_gp_targets.pth"))
        torch.save(gp_model.state_dict(), os.path.join(NAME_RESULT_DIR, f"{name}_gp_model.pth"))
        params = {k: v.cpu().numpy().tolist() for k, v in gp_model.state_dict().items()}
        with open(os.path.join(NAME_RESULT_DIR, f"{name}_gp_params.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4))
        trained_gps.append(gp_model)

        scaler = 1
        if name in ["omega_f", "dot_beta"]:
            scaler = 180/np.pi

        # Select GP dataset for evaluation by farthest point sampling:
        normalized_data = (filtered_data - np.mean(filtered_data, axis=0)) / (np.std(filtered_data, axis=0))
        # normalized_data = (filtered_data - np.min(filtered_data, axis=0)) / (np.max(filtered_data, axis=0) - np.min(filtered_data, axis=0))
        selected_idx = farthest_point_sampling(normalized_data[:, input_indices], num_samples=int(500))
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
                nominal = -(config.avant_lr * data[:, 5] + data[:, 6] * np.sin(data[:, 3])) / (config.avant_lf * np.cos(data[:, 3]) + config.avant_lr)
            elif name == "v_f":
                nominal = 3.5*data[:, 8] - 0.085
            elif name in ["dot_beta", "dot_dot_beta"]:
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
            mean += nominal

            fig, ax = plt.subplots(1, figsize=(2000 / 100, 1000 / 100))
            fig.tight_layout()

            ax.plot(scaler * data[:, output_i], label="target")
            ax.plot(scaler * (nominal), label="nominal model")
            ax.plot(scaler * (mean), label="predicted")
            ax.fill_between(np.arange(len(mean)),
                scaler * (mean - 2*gp_outputs.variance.sqrt()[:, 0].cpu().numpy()),
                scaler * (mean + 2*gp_outputs.variance.sqrt()[:, 0].cpu().numpy()),
                alpha=0.5, label="95 uncertainty interval"
            )      

            nominal_error = np.sum((data[:, output_i] - nominal)**2)   
            gp_error = np.sum((data[:, output_i] - mean)**2)     
            nominal_errors.append(nominal_error)
            gp_errors.append(gp_error)

            if name == "v_f":
                gas_table = np.array([0.0, 0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                speed_table = np.array([0.0, 0.15, 0.16, 0.25, 0.4, 0.9, 1.35, 1.66, 2.1, 2.55, 2.95, 3.05, 3.15])
                speed_value = interpolate.interp1d(gas_table, speed_table, 'linear', bounds_error=False, fill_value=(0.0, 1.0))
                signs = np.sign(data[:, 8])
                ax.plot(signs * speed_value(np.abs(data[:, 8])), label="Current lookup table")

            ax.legend()
            plt.savefig(os.path.join(FILE_RESULT_DIR, f"{file}_{name}_fit.png"))
            plt.close()

        nominal_errors = np.asarray(nominal_errors)
        gp_errors = np.asarray(gp_errors)
        print(f"{name}:\nNominal squared error: {nominal_errors.mean()} (+/- {2*nominal_errors.std()})\nGP squared error: {gp_errors.mean()} (+/- {2*gp_errors.std()})")
        
        means = [nominal_errors.mean(), gp_errors.mean()]
        errors = [2 * nominal_errors.std(), 2 * gp_errors.std()]
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.title(f"{name} mean squared error")
        fig.tight_layout()
        positions = np.arange(len(means))
        bars = ax.bar(positions, means, yerr=errors, capsize=10, tick_label=["Nominal", "GP corrected"])

        plt.savefig(os.path.join(RESULTS_DIR, f"{name}_squared_errors.png"))
        plt.close()

