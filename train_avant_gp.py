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
from sparse_to_dense_reward.avant_dynamics import AvantDynamics, OldDynamics

DATA_SOURCE_DIR = 'avant_identification_data/'
current_time = datetime.now().strftime("%H_%M_%S")
RESULTS_DIR = f'avant_identification_results/{current_time}/'
os.makedirs(RESULTS_DIR, exist_ok=True)

data_idx_deriv = [
    # Inputs
    (1, False, "x"), 
    (2, False, "y"), 
    (3, False, "theta"), 
    (4, False, "beta"),   
    (5, False, "omega"),     
    (6, False, "dot_beta"),   
    (7, False, "v_f"),        
    (8, False, "steer"),      
    (9, False, "gas"),        
    # Outputs
    (6, True, "dot_dot_beta"),   
    (7, True, "a_f"),   
]

if __name__ == '__main__':
    original_data, filtered_data = [], []
    per_file_eval_datas = []

    for file in os.listdir(DATA_SOURCE_DIR):
        if ".csv" not in file:
            continue
        print(f"Processing file: {file}")
        data_batch = np.loadtxt(os.path.join(DATA_SOURCE_DIR, file), skiprows=1, delimiter=",")
        timestamps, original, filtered = filter_data_batch(data_batch, data_idx_deriv, os.path.join(DATA_SOURCE_DIR, file[:-4]))
        original_data.extend(original)
        filtered_data.extend(filtered)
        per_file_eval_datas.append((file, timestamps, filtered))

    original_data = np.asarray(original_data)
    filtered_data = np.asarray(original_data)

    print(f"Num data: {len(filtered_data)}")

    # Train the GPs independently:
    gp_datas = [
        ("omega_f", [3, 5, 6, 7], 4),         # inputs: beta, dot beta and v_f
        ("v_f", [3, 8], 6),                   # inputs: beta and gas
        ("dot_dot_beta", [3, 5, 6, 7], 9)     # inputs: beta, dot_beta, v_f and u_steer
    ]
    trained_gps = []
    for name, input_indices, output_i in gp_datas:
        # normalized_data = (filtered_data - np.mean(filtered_data, axis=0)) / (np.std(filtered_data, axis=0))
        normalized_data = (filtered_data - np.min(filtered_data, axis=0)) / (np.max(filtered_data, axis=0) - np.min(filtered_data, axis=0))
        selected_idx = farthest_point_sampling(normalized_data[:, input_indices], num_samples=int(333))
        plot_sample_data(filtered_data, selected_idx, os.path.join(RESULTS_DIR, f"{name}_samples.png"))

        nominal = torch.zeros(len(filtered_data))
        if name == "omega_f":
            nominal = AvantDynamics.get_nominal_omega(filtered_data[:, 3], filtered_data[:, 5], filtered_data[:, 6])
        elif name == "v_f":
            nominal = AvantDynamics.get_nominal_v(filtered_data[:, 8])

        gp_inputs = torch.from_numpy(filtered_data[selected_idx][:, input_indices]).to(torch.float).cuda()
        gp_targets = torch.from_numpy(filtered_data[selected_idx, output_i]).to(torch.float).cuda() - nominal[selected_idx].to(torch.float).cuda()
        gp_model = GPModel(gp_inputs, gp_targets, train_epochs=100, train_lr=1e-1)
        torch.save(gp_inputs, os.path.join(RESULTS_DIR, f"{name}_gp_inputs.pth"))
        torch.save(gp_targets, os.path.join(RESULTS_DIR, f"{name}_gp_targets.pth"))
        torch.save(gp_model.state_dict(), os.path.join(RESULTS_DIR, f"{name}_gp_model.pth"))
        params = {k: v.cpu().numpy().tolist() for k, v in gp_model.state_dict().items()}
        with open(os.path.join(RESULTS_DIR, f"{name}_gp_params.json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4))
        trained_gps.append(gp_model)

        scaler = 1
        if name in ["omega_f", "dot_dot_beta"]:
            scaler = 180/np.pi

        # Evaluate the trained model for each data file:
        for file, timestamps, data in per_file_eval_datas:
            gp_eval_inputs = torch.from_numpy(data[:, input_indices]).to(torch.float).cuda()
            with torch.no_grad():
                gp_outputs = gp_model(gp_eval_inputs)
            mean = gp_outputs.mean.cpu().numpy()[:, 0]
            nominal = torch.zeros(len(data))
            if name == "omega_f":
                nominal = AvantDynamics.get_nominal_omega(data[:, 3], data[:, 5], data[:, 6])
            elif name == "v_f":
                nominal = AvantDynamics.get_nominal_v(data[:, 8])
            mean += nominal.numpy()

            fig, ax = plt.subplots(1, figsize=(2000 / 100, 1000 / 100))
            fig.tight_layout()

            ax.plot(scaler * data[:, output_i], label="target")
            ax.plot(scaler * (mean), label="predicted")
            ax.fill_between(np.arange(len(mean)),
                scaler * (mean - 2*gp_outputs.variance.sqrt()[:, 0].cpu().numpy()),
                scaler * (mean + 2*gp_outputs.variance.sqrt()[:, 0].cpu().numpy()),
                alpha=0.5, label="95 uncertainty interval"
            )            
            
            if name == "omega_f":
                ax.plot(scaler * nominal, label="Reza's paper")

            if name == "v_f":
                gas_table = np.array([0.0, 0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                speed_table = np.array([0.0, 0.15, 0.16, 0.25, 0.4, 0.9, 1.35, 1.66, 2.1, 2.55, 2.95, 3.05, 3.15])
                speed_value = interpolate.interp1d(gas_table, speed_table, 'linear', bounds_error=False, fill_value=(0.0, 1.0))
                signs = np.sign(data[:, 8])
                ax.plot(signs * speed_value(np.abs(data[:, 8])), label="Current lookup table")

            ax.legend()
            plt.savefig(os.path.join(RESULTS_DIR, f"{file}_{name}_fit.png"))
            plt.close()

    # Simulate the system according to the controls in our training data file(s):
    avant_model = AvantDynamics(config.sample_rate, trained_gps[0], trained_gps[1], trained_gps[2], "cuda:0")
    old_avant_model = OldDynamics(config.sample_rate)
    for file, timestamps, data in per_file_eval_datas:
        KEYS = ['x', 'y', 'theta', 'beta', 'omega', 'dot_beta', 'vx']
        times = []
        predictions = []
        old_prediction = []
    
        state_t = data[0, :7].copy()
        for data_t in data:
            predictions.append(state_t)
            model_state = torch.from_numpy(np.r_[state_t[:7], np.zeros(3)]).unsqueeze(0).to(torch.float).cuda()
            model_input = torch.from_numpy(data_t[7:9]).unsqueeze(0).to(torch.float).cuda()
            with torch.no_grad():
                model_output = avant_model.discrete_dynamics_fun(model_state, model_input)
            state_t = model_output[0, :7].cpu().numpy()
        
        # Propagate the old model for comparision:
        state_t = data[0, :4].copy()
        for data_t in data:
            old_prediction.append(state_t)
            model_state = torch.from_numpy(state_t).unsqueeze(0).to(torch.float).cuda()
            model_input = torch.from_numpy(data_t[7:9]).unsqueeze(0).to(torch.float).cuda()
            with torch.no_grad():
                model_output = old_avant_model.discrete_dynamics_fun(model_state, model_input)
            state_t = model_output[0, :4].cpu().numpy()

        # header = 'timestamp,' + ','.join(KEYS)
        predictions = np.asarray(predictions)
        old_prediction = np.asarray(old_prediction)
        # np.savetxt(os.path.join(RESULTS_DIR, f'{file}.csv'), np.column_stack((timestamps, predictions)), delimiter=',', header=header, comments='')

        fig, ax = plt.subplots(1, figsize=(1000 / 100, 1000 / 100))
        fig.tight_layout()
        ax.scatter(data[:, 0], data[:, 1], label="odometry")
        ax.scatter(old_prediction[:, 0], old_prediction[:, 1], label="(old) predicted")
        ax.scatter(predictions[:, 0], predictions[:, 1], label="(new) predicted")
        ax.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{file}_map.png"))
        plt.close()