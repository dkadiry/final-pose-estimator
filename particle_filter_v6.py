import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='particle_filter.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

np.random.seed(42)

class ParticleFilter:
    np.random.seed(42)
    def __init__(self, num_particles, initial_state, initial_timestamp, process_noise_std, measurement_noise_std):
        """
        Initialize the Particle Filter.

        Parameters:
        - num_particles (int): Number of particles.
        - initial_state (list or np.array): Initial relative pose [Δx_total, Δz_total, Δyaw_total].
        - initial_timestamp (int or float): Starting timestamp (e.g., in milliseconds).
        - process_noise_std (list or tuple): Standard deviations for process noise [std_v, std_omega].
        - measurement_noise_std (list or tuple): Standard deviations for measurement noise [std_tx, std_tz, std_yaw].
        """
        self.num_particles = num_particles
        self.particles = np.tile(initial_state, (num_particles, 1))  # Shape: (num_particles, 3)
        self.weights = np.ones(num_particles) / num_particles  # Initialize weights uniformly
        self.current_timestamp = initial_timestamp
        self.process_noise_std = np.array(process_noise_std)  # [std_v, std_omega]
        self.measurement_noise_std = np.array(measurement_noise_std)  # [std_tx, std_tz, std_yaw]

        logging.info(f"Initialized ParticleFilter with {num_particles} particles.")

    def predict(self, velocity_data, next_timestamp):
        """
        Predict the state of each particle based on velocity commands up to the next timestamp.

        Parameters:
        - velocity_data (pd.DataFrame): DataFrame containing velocity commands with columns ['timestamp', 'x_linear', 'z_angular'].
        - next_timestamp (int or float): The timestamp up to which to apply velocity commands.
        """
        # Select velocity commands between current and next timestamps
        interval_velocity = velocity_data[
            (velocity_data['timestamp'] >= self.current_timestamp) &
            (velocity_data['timestamp'] <= next_timestamp)
        ].sort_values('timestamp')

        for _, vel in interval_velocity.iterrows():
            v = vel['x_linear']       # Forward linear velocity (m/s)
            omega = vel['z_angular']  # Angular velocity (rad/s)
            dt = (vel['timestamp'] - self.current_timestamp) / 1000.0  # Convert ms to seconds

            self.motion_model_vectorized(v, omega, dt)
            self.current_timestamp = vel['timestamp']

            logging.debug(f"Predicted particles with v={v}, omega={omega}, dt={dt}")

    def motion_model_vectorized(self, v, omega, dt):
        """
        Vectorized motion model to update all particles based on velocity commands.

        Parameters:
        - v (float): Forward linear velocity (m/s).
        - omega (float): Angular velocity (rad/s).
        - dt (float): Time interval in seconds.
        """

        # Add process noise
        v_noise = np.random.normal(0, self.process_noise_std[0], self.num_particles)
        omega_noise = np.random.normal(0, self.process_noise_std[1], self.num_particles)

        # Update X, Z, and Yaw
        # Compute the change in position based on current yaw
        delta_x = (v + v_noise) * dt * np.sin(self.particles[:, 2])
        delta_z = (v + v_noise) * dt * np.cos(self.particles[:, 2])

        # Compute the change in yaw
        delta_yaw = (omega + omega_noise) * dt

        # Update particles with relative changes
        self.particles[:, 0] += delta_x
        self.particles[:, 1] += delta_z
        self.particles[:, 2] += delta_yaw

        # Normalize yaw to [-pi, pi]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        logging.debug("Motion model updated particle states.")

    def update(self, absolute_measurement, is_scaled):
        """
        Update the weights of each particle based on the absolute pose measurement.

        Parameters:
        - Absolute_measurement (np.array): Absolute measurement array [x_total, z_total, yaw_total].
        - is_scaled (int): Flag indicating measurement scaling (1 for full, 0 for yaw only).
        """
        if is_scaled == 1:
            x_meas, z_meas, yaw_meas = absolute_measurement
            logging.debug("Updating weights with full absolute measurement: x, z, yaw.")

            # Compute measurement likelihood
            error_x = self.particles[:, 0] - x_meas
            error_z = self.particles[:, 1] - z_meas
            error_yaw = self.particles[:, 2] - yaw_meas
            error_yaw = (error_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize

            # Calculate probability densities
            prob_x = self.simplified_likelihood(error_x, self.measurement_noise_std[0])
            prob_z = self.simplified_likelihood(error_z, self.measurement_noise_std[1])
            prob_yaw = self.simplified_likelihood(error_yaw, self.measurement_noise_std[2])

            # Update weights
            self.weights *= prob_x * prob_z * prob_yaw

        else:
            yaw_meas = absolute_measurement[2]
            logging.debug("Updating weights with yaw only cumulative measurement.")

            # Compute measurement likelihood
            error_yaw = self.particles[:, 2] - yaw_meas
            error_yaw = (error_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize

            # Calculate probability densities
            prob_yaw = self.simplified_likelihood(error_yaw, self.measurement_noise_std[2])

            # Update weights
            self.weights *= prob_yaw

        # Normalize weights
        self.weights += 1e-300  # Prevent division by zero
        self.weights /= np.sum(self.weights)

        logging.debug("Weights updated and normalized.")

        # Resample particles based on updated weights
        self.resample()

    @staticmethod
    def simplified_likelihood(error, std):
        """
        Calculate simplified Gaussian probability density function (without normalization).

        Parameters:
        - error (np.array): Error term.
        - std (float): Standard deviation.

        Returns:
        - prob (np.array): Probability densities.
        """
        return np.exp(-0.5 * (error / std) ** 2)

    @staticmethod
    def gaussian(error, std):
        """
        Calculate Gaussian probability density function.

        Parameters:
        - error (np.array): Error term.
        - std (float): Standard deviation.

        Returns:
        - prob (np.array): Probability densities.
        """
        return (1.0 / (np.sqrt(2.0 * np.pi) * std)) * np.exp(-0.5 * (error / std) ** 2)

    def resample(self):
        """
        Resample particles based on their weights using systematic resampling.
        """
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly one
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        pointers = start + step * np.arange(self.num_particles)
        indexes = np.searchsorted(cumulative_sum, pointers)

        # Resample particles and reset weights
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)

        logging.debug("Particles resampled based on weights.")

    def estimate(self):
        """
        Estimate the current absolute pose as the weighted average of the particles.

        Returns:
        - mean_pose (np.array): Estimated absolute pose [x_total, z_total, yaw_total].
        """
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_z = np.average(self.particles[:, 1], weights=self.weights)

        # Handle yaw estimation with circular statistics
        sin_yaw = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_yaw = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        mean_yaw = np.arctan2(sin_yaw, cos_yaw)

        mean_pose = np.array([mean_x, mean_z, mean_yaw])

        logging.debug(f"Estimated absolute pose: x_total={mean_x}, z_total={mean_z}, yaw_total={mean_yaw}")
        return mean_pose

def calculate_rmse(synchronized_data):
    """
    Calculate RMSE between PF estimates and ground truth relative poses.

    Parameters:
    - synchronized_data (pd.DataFrame): DataFrame with ['x_est', 'z_est', 'yaw_est', 'tx', 'tz', 'yaw']

    Returns:
    - rmse_x, rmse_z, rmse_yaw
    """
    rmse_x = np.sqrt(np.mean((synchronized_data['x_est'] - synchronized_data['tx'])**2))
    rmse_z = np.sqrt(np.mean((synchronized_data['z_est'] - synchronized_data['tz'])**2))
    angle_diff = np.mod(synchronized_data['yaw_est'] - synchronized_data['yaw'] + np.pi, 2 * np.pi) - np.pi
    rmse_yaw = np.sqrt(np.mean(angle_diff**2))
    return rmse_x, rmse_z, rmse_yaw

def evaluate_parameters(params, velocity_data, absolute_measurements, ground_truth_df):
    """
    Evaluate a single parameter set.

    Parameters:
    - params: Tuple of (std_v, std_omega)
    - velocity_data: DataFrame with velocity commands
    - cumulative_measurements: DataFrame with cumulative measurements
    - ground_truth_df: DataFrame with ground truth relative poses ['pose_timestamp', 'tx', 'tz', 'yaw']

    Returns:
    - (std_v, std_omega, total_rmse)
    """
    std_v, std_omega = params
    logging.info(f"Evaluating std_v={std_v}, std_omega={std_omega}")

    # Initialize PF
    pf = ParticleFilter(
        num_particles=1000,
        initial_state=[0.0, 0.0, 0.0],
        initial_timestamp=1717696051836,
        process_noise_std=[std_v, std_omega],
        measurement_noise_std=[4.430752218268109, 4.567227516109891, 1.1157471753680754] # Measurement noise computed empirically
    )

    # Initialize storage for estimates
    estimates = []

    for i, measurement in absolute_measurements.iterrows():

        # Prediction
        pf.predict(velocity_data, next_timestamp=measurement['pose_timestamp'])

        # Prepare absolute measurement
        absolute_measurement = np.array([
            measurement['tx'],
            measurement['tz'],
            measurement['yaw']
        ])
        is_scaled = measurement['is_scaled']

        # Update
        pf.update(absolute_measurement, is_scaled=is_scaled)

        # Estimate absolute pose after update
        current_estimate = pf.estimate()
        estimates.append({
            'pose_timestamp': measurement['pose_timestamp'],
            'x_est': current_estimate[0],
            'z_est': current_estimate[1],
            'yaw_est': current_estimate[2]
        })
        

    # Convert estimates to DataFrame
    estimates_df = pd.DataFrame(estimates)

    # Merge with ground truth
    synchronized_data = pd.merge(
        estimates_df,
        ground_truth_df,
        on='pose_timestamp',
        how='inner',
        suffixes=('_est', '')
    )

    if synchronized_data.empty:
        logging.warning(f"No synchronized data for std_v={std_v}, std_omega={std_omega}.")
        return (std_v, std_omega, float('inf'))

    # Calculate RMSE
    rmse_x, rmse_z, rmse_yaw = calculate_rmse(synchronized_data)
    total_rmse = (0.6 * rmse_x) + (0.3 * rmse_z) + (0.1 * rmse_yaw)  # Adjust weighting if necessary

    logging.info(f"std_v={std_v}, std_omega={std_omega} => RMSE: {total_rmse:.4f}")
    return (std_v, std_omega, total_rmse)

def grid_search_parallel(std_v_values, std_omega_values, velocity_data, absolute_measurements, ground_truth_df):
    """
    Perform parallel Grid Search to find the best std_v and std_omega.

    Parameters:
    - std_v_values: List of std_v values to test
    - std_omega_values: List of std_omega values to test
    - velocity_data: DataFrame with velocity commands
    - cumulative_measurements: DataFrame with cumulative measurements
    - ground_truth_df: DataFrame with ground truth relative poses

    Returns:
    - best_params: Tuple (std_v, std_omega)
    - best_rmse: Corresponding RMSE value
    """
    from itertools import product
    import multiprocessing as mp

    parameter_grid = list(product(std_v_values, std_omega_values))
    pool = mp.Pool(processes=mp.cpu_count())

    results = []
    for params in parameter_grid:
        results.append(pool.apply_async(evaluate_parameters, args=(params, velocity_data, absolute_measurements, ground_truth_df)))

    pool.close()
    pool.join()

    # Collect results
    evaluated_params = [res.get() for res in results]

    # Filter out unsuccessful evaluations
    evaluated_params = [res for res in evaluated_params if res[2] != float('inf')]

    if not evaluated_params:
        logging.error("No successful evaluations were performed. Check your data and parameters.")
        raise ValueError("No successful evaluations were performed. Check your data and parameters.")

    # Find the best parameters with the lowest RMSE
    best_result = min(evaluated_params, key=lambda x: x[2])
    best_std_v, best_std_omega, best_rmse = best_result

    logging.info(f"Best Parameters: std_v={best_std_v}, std_omega={best_std_omega} with RMSE={best_rmse:.4f}")
    return (best_std_v, best_std_omega), best_rmse

if __name__ == "__main__":
    # Load your datasets with correct file paths
    ground_truth_df = pd.read_csv("Data\Ground_truths\euler_RFC_ground_truth.csv")  # ['pose_timestamp', 'tx', 'tz', 'yaw', ...]
    velocity_data = pd.read_csv("Data\Ground_truths\Copy of Joystick Command Data.csv")  # ['timestamp', 'x_linear', 'z_angular']
    absolute_measurements = pd.read_csv("Pose_Results\Final_Approach_results\Raw Poses\euler_raw_absolute_poses_approach5_5.csv") # ['pose_timestamp', 'tx', 'tz', 'yaw', 'is_scaled', ...]

    # Define the cutoff timestamp
    cutoff_timestamp = 1717696314266
    initial_timestamp = 1717696051836

    # Filter rows to cutoff timestamp
    ground_truth_df_filtered = ground_truth_df[(ground_truth_df["pose_timestamp"] > initial_timestamp) & (ground_truth_df["pose_timestamp"] <= cutoff_timestamp)].reset_index(drop=True)
    velocity_data_filtered = velocity_data[velocity_data["timestamp"] <= cutoff_timestamp].reset_index(drop=True)
    absolute_measurements_filtered = absolute_measurements[(absolute_measurements["pose_timestamp"] > initial_timestamp) & (absolute_measurements["pose_timestamp"] <= cutoff_timestamp)].reset_index(drop=True)

    # Verify alignment 
    # Ensure ground_truth_df has the same number of measurements and matching pose_timestamps
    aligned = absolute_measurements_filtered['pose_timestamp'].equals(ground_truth_df_filtered['pose_timestamp'])
    if not aligned:
        logging.error("Timestamps are not aligned after filtering.")
        raise ValueError("Timestamps are not aligned between measurements and ground truth.")
    else:
        logging.info("Timestamps btwn Ground truth and Absolute Measurements are aligned after filtering.")

    # Sort measurements by timestamp
    absolute_measurements_filtered = absolute_measurements_filtered.sort_values('pose_timestamp').reset_index(drop=True)
 
    # Define parameter ranges for Grid Search
    std_v_values = [0.01, 0.02, 0.05, 0.07, 0.1]      
    std_omega_values = [0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]        

    # Perform Grid Search to find the best parameters
    best_params, best_rmse = grid_search_parallel(
        std_v_values=std_v_values,
        std_omega_values=std_omega_values,
        velocity_data=velocity_data_filtered,
        absolute_measurements=absolute_measurements_filtered,
        ground_truth_df=ground_truth_df_filtered
    )

    print(f"Best Parameters: std_v={best_params[0]}, std_omega={best_params[1]} with RMSE={best_rmse:.4f}")

    # Initialize PF with best parameters
    initial_state = [0.0, 0.0, 0.0]  # [x_total, z_total, yaw_total]
    

    # Fixed noise parameters (to run without tuning)
    std_v_fixed = 0.01
    std_omega_fixed = 0.02

    pf_best = ParticleFilter(
        num_particles=1000,
        initial_state=initial_state,  # Initial absolute pose
        initial_timestamp=initial_timestamp,
        #process_noise_std=[std_v_fixed, std_omega_fixed],
        process_noise_std=list(best_params),
        measurement_noise_std= [4.430752218268109, 4.567227516109891, 1.1157471753680754]   # Measurement noise empirically calculated
    )

    # Initialize storage for cumulative and incremental estimates
    estimates_best_filtered = []

    for i, measurement in absolute_measurements_filtered.iterrows():

        # Prediction
        pf_best.predict(velocity_data_filtered, next_timestamp=measurement['pose_timestamp'])
        
        # Prepare absolute measurement
        absolute_measurement = np.array([
            measurement['tx'],
            measurement['tz'],
            measurement['yaw']
        ])
        is_scaled = measurement['is_scaled']
        
        # Update
        pf_best.update(absolute_measurement, is_scaled=is_scaled)
        
        # Estimate absolute pose after update
        current_estimate_best_filtered = pf_best.estimate()
        estimates_best_filtered.append({
            'pose_timestamp': measurement['pose_timestamp'],
            'x_est': current_estimate_best_filtered[0],
            'z_est': current_estimate_best_filtered[1],
            'yaw_est': current_estimate_best_filtered[2]
        })
        
    # Convert estimates to DataFrame
    estimates_best_df_filtered = pd.DataFrame(estimates_best_filtered)

    #print(estimates_best_df_filtered.head(5))

    # Merge with ground truth
    synchronized_data_best_filtered = pd.merge(
        estimates_best_df_filtered,
        ground_truth_df_filtered,
        on='pose_timestamp',
        how='inner',
        suffixes=('_est', '')
    ) 

    # Calculate final RMSE
    final_rmse_x, final_rmse_z, final_rmse_yaw = calculate_rmse(synchronized_data_best_filtered)
    print(f"Final RMSE - X: {final_rmse_x:.4f} m, Z: {final_rmse_z:.4f} m, Yaw: {final_rmse_yaw:.4f} rad")

    # Plot trajectories for visual assessment
    plt.figure(figsize=(10, 8))
    plt.plot(synchronized_data_best_filtered['tx'], synchronized_data_best_filtered['tz'],
             label='Ground Truth', linewidth=2)
    plt.plot(synchronized_data_best_filtered['x_est'], synchronized_data_best_filtered['z_est'],
             label='PF Estimate', linestyle='--')
    plt.plot(absolute_measurements_filtered['tx'], absolute_measurements_filtered['tz'],
             label='Measurements', linestyle=':')
    plt.xlabel('X Position (Lateral) [m]')
    plt.ylabel('Z Position (Forward) [m]')
    plt.title('Trajectory Comparison: Ground Truth vs. PF Estimate vs Measurements')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the synchronized data with estimates and ground truth
    #synchronized_data_best_filtered.to_csv("V6_Updated_PF_Best_Estimates_With_Ground_Truth_&_Filtered_Timestamps.csv", index=False)


    # Filter Absolute Measurements for final run (Ignore first measurment which is origin)
    absolute_measurements_final = absolute_measurements[absolute_measurements["pose_timestamp"] > initial_timestamp].reset_index(drop=True)

    # Sort measurements by timestamp
    absolute_measurements_final = absolute_measurements_final.sort_values('pose_timestamp').reset_index(drop=True)

    # Initialize Final PF with best parameters
    final_initial_state = [0.0, 0.0, 0.0]  # [x_total, z_total, yaw_total]

    pf_final = ParticleFilter(
        num_particles=1000,
        initial_state=final_initial_state,  # Initial absolute pose
        initial_timestamp=initial_timestamp,
        process_noise_std=list(best_params),
        measurement_noise_std= [4.430752218268109, 4.567227516109891, 1.1157471753680754]   # Measurement noise empirically calculated
    )

    # Initialize storage for best estimates
    estimates_best = []

    for i, measurement in absolute_measurements_final.iterrows():

        # Prediction
        pf_final.predict(velocity_data, next_timestamp=measurement['pose_timestamp'])
        
        # Prepare absolute measurement
        absolute_measurement = np.array([
            measurement['tx'],
            measurement['tz'],
            measurement['yaw']
        ])
        is_scaled = measurement['is_scaled']
        
        # Update
        pf_final.update(absolute_measurement, is_scaled=is_scaled)
        
        # Estimate absolute pose after update
        current_estimate_best = pf_final.estimate()
        estimates_best.append({
            'pose_timestamp': measurement['pose_timestamp'],
            'x_est': current_estimate_best[0],
            'z_est': current_estimate_best[1],
            'yaw_est': current_estimate_best[2]
        })
        
    # Convert estimates to DataFrame
    estimates_best_df = pd.DataFrame(estimates_best)

    #print(estimates_best_df.head(5))

    # Merge x_est, z_est, yaw_est with y, pitch, and roll from raw absolute measurements
    
    #Select Specific columns
    selected_absolute_df = absolute_measurements[['pose_timestamp', 'ty', 'pitch', 'roll']]
    selected_estimates_best_df = estimates_best_df[['pose_timestamp', 'x_est', 'z_est', 'yaw_est']]

    merged_data_df = pd.merge(selected_estimates_best_df, selected_absolute_df, on='pose_timestamp', how="inner")

    # Re order the columns
    desired_order = ['pose_timestamp', 'x_est', "ty", "z_est", 'pitch', 'yaw_est', 'roll']

    final_df = merged_data_df[desired_order]

    final_df.rename(columns={
        'x_est': "tx",
        'z_est': "tz",
        "yaw_est": 'yaw'
    },inplace=True)

    first_row = absolute_measurements.iloc[0][['pose_timestamp', 'tx', 'ty', 'tz', 'pitch', 'yaw', 'roll']]

    # Convert the selected columns to a DataFrame row 
    first_row_df = pd.DataFrame([first_row], columns=['pose_timestamp', 'tx', 'ty', 'tz', 'pitch', 'yaw', 'roll'])

    final_df = pd.concat([first_row_df, final_df], ignore_index=True)

    final_df = final_df.sort_values('pose_timestamp').reset_index(drop=True)

    # Save the synchronized data with estimates and ground truth
    #final_df.to_csv("V6_Updated_PF_Best_Estimates_Absolute_RFC_Pose.csv", index=False)
