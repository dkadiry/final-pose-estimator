import pandas as pd
import numpy as np
#from sklearn.metrics import mean_squared_error

estimated_poses_df = pd.read_csv("V6_Updated_PF_Best_Estimates_Absolute_RFC_Pose.csv")
ground_truth_df = pd.read_csv("Data\Ground_truths\euler_RFC_ground_truth.csv")

def mean_squared_error(true, pred):
    return np.mean((np.array(true) - np.array(pred)) ** 2)

cutoff_timestamp = 1717696314266

# Filter rows to cutoff timestamp
estimated_poses_df = estimated_poses_df[estimated_poses_df["pose_timestamp"] <= cutoff_timestamp]
ground_truth_df = ground_truth_df[ground_truth_df["pose_timestamp"] <= cutoff_timestamp]

tx_values_est = []
tx_values_gt = []
tz_values_est = []
tz_values_gt = []
ty_values_gt = []
ty_values_est = []
yaw_values_est = []
yaw_values_gt = []
pitch_values_est = []
pitch_values_gt = []
roll_values_est = []
roll_values_gt = []

for _, est_row in estimated_poses_df.iterrows():
    timestamp = est_row["pose_timestamp"]

    #Find the corresponding row in ground truth
    gt_row = ground_truth_df[ground_truth_df["pose_timestamp"] == timestamp]
    if gt_row.empty:
        continue

    # Extract rotation angle values
    yaw_values_est.append(est_row["yaw"])
    yaw_values_gt.append(gt_row['yaw'].values[0])
    pitch_values_est.append(est_row['pitch'])
    pitch_values_gt.append(gt_row['pitch'].values[0])
    roll_values_est.append(est_row['roll'])
    roll_values_gt.append(gt_row['roll'].values[0])

    #Extract translation values where is_scaled = 1 for raw poses, but all for particl filtered poses
    #if est_row['is_scaled'] == 1: #Uncomment when evaluating raw poses pre-particle filter and move the extraction block into the if
    # Append tx values
    tx_values_est.append(est_row["tx"])
    tx_values_gt.append(gt_row['tx'].values[0])

    #Append ty values
    ty_values_est.append(est_row['ty'])
    ty_values_gt.append(gt_row['ty'].values[0])

    # Append tz values
    tz_values_est.append(est_row['tz'])
    tz_values_gt.append(gt_row['tz'].values[0])

# Function to compute filtered RMSE
# Function to compute filtered RMSE based on IQR
def filtered_rmse_iqr(gt_values, est_values):
    errors = np.abs(np.array(est_values) - np.array(gt_values))
    
    # Compute IQR
    q1, q3 = np.percentile(errors, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    # Filter out errors outside the IQR-defined bounds
    filtered_gt = [gt for gt, err in zip(gt_values, errors) if lower_bound <= err <= upper_bound]
    filtered_est = [est for est, err in zip(est_values, errors) if lower_bound <= err <= upper_bound]
    
    return np.sqrt(mean_squared_error(filtered_gt, filtered_est)) if filtered_gt else None

def calculate_filtered_mape(gt_values, est_values):
    errors = np.abs(np.array(gt_values) - np.array(est_values))
    
    # Compute IQR
    q1, q3 = np.percentile(errors, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    # Filter out errors outside the IQR-defined bounds
    filtered_gt = [gt for gt, err in zip(gt_values, errors) if lower_bound <= err <= upper_bound]
    filtered_est = [est for est, err in zip(est_values, errors) if lower_bound <= err <= upper_bound]

    filtered_gt = np.array(filtered_gt)
    filtered_est = np.array(filtered_est)

    epsilon = 1e-10
    mape = np.mean(np.abs((filtered_gt - filtered_est) / (filtered_gt + epsilon))) * 100 if len(filtered_gt) > 0 else None
    return mape

def calculate_mape(true_values, estimated_values):
    true_values = np.array(true_values)
    estimated_values = np.array(estimated_values)
    
    # Avoid division by zero by adding a small epsilon to true values where they are zero
    epsilon = 1e-10
    mape = np.mean(np.abs((true_values - estimated_values) / (true_values + epsilon))) * 100
    return mape

# Calculate MAPE for each pose component
tx_mape = calculate_mape(tx_values_gt, tx_values_est)
ty_mape = calculate_mape(ty_values_gt, ty_values_est)
tz_mape = calculate_mape(tz_values_gt, tz_values_est)
yaw_mape = calculate_mape(yaw_values_gt, yaw_values_est)
pitch_mape = calculate_mape(pitch_values_gt, pitch_values_est)
roll_mape = calculate_mape(roll_values_gt, roll_values_est)

# Find RMSE for all pose components
tx_rmse = np.sqrt(mean_squared_error(tx_values_gt, tx_values_est)) if tx_values_gt else None
ty_rmse = np.sqrt(mean_squared_error(ty_values_gt, ty_values_est)) if ty_values_gt else None
tz_rmse = np.sqrt(mean_squared_error(tz_values_gt, tz_values_est)) if tz_values_gt else None
yaw_rmse = np.sqrt(mean_squared_error(yaw_values_gt, yaw_values_est)) if yaw_values_gt else None
pitch_rmse = np.sqrt(mean_squared_error(pitch_values_gt, pitch_values_est)) if pitch_values_gt else None
roll_rmse = np.sqrt(mean_squared_error(roll_values_gt, roll_values_est)) if roll_values_gt else None

# Find Normalized RMSE for all pose components
#tx_nrmse = tx_rmse / (max(tx_values_gt) - min(tx_values_gt))
#ty_nrmse = ty_rmse / (max(ty_values_gt) - min(ty_values_gt))
#tz_nrmse = tz_rmse / (max(tz_values_gt) - min(tz_values_gt))
#yaw_nrmse = yaw_rmse / (max(yaw_values_gt) - min(yaw_values_gt))
#pitch_nrmse = pitch_rmse / (max(pitch_values_gt) - min(pitch_values_gt))
#roll_nrmse = roll_rmse / (max(roll_values_gt) - min(roll_values_gt))


# Calculate Median Absolute Error (MedAE)
tx_medae = np.median(np.abs(np.array(tx_values_est) - np.array(tx_values_gt)))
ty_medae = np.median(np.abs(np.array(ty_values_est) - np.array(ty_values_gt)))
tz_medae = np.median(np.abs(np.array(tz_values_est) - np.array(tz_values_gt)))
yaw_medae = np.median(np.abs(np.array(yaw_values_est) - np.array(yaw_values_gt)))
pitch_medae = np.median(np.abs(np.array(pitch_values_est) - np.array(pitch_values_gt)))
roll_medae = np.median(np.abs(np.array(roll_values_est) - np.array(roll_values_gt)))

# Calculate Filtered RMSE with threshold of 0.5 as example
tx_filtered_rmse = filtered_rmse_iqr(tx_values_gt, tx_values_est)
ty_filtered_rmse = filtered_rmse_iqr(ty_values_gt, ty_values_est)
tz_filtered_rmse = filtered_rmse_iqr(tz_values_gt, tz_values_est)
yaw_filtered_rmse = filtered_rmse_iqr(yaw_values_gt, yaw_values_est)
pitch_filtered_rmse = filtered_rmse_iqr(pitch_values_gt, pitch_values_est)
roll_filtered_rmse = filtered_rmse_iqr(roll_values_gt, roll_values_est)

# Find Normalized Filtered RMSE for all pose components
#tx_filtered_nrmse = tx_filtered_rmse / (max(tx_values_gt) - min(tx_values_gt))
#ty_filtered_nrmse = ty_filtered_rmse / (max(ty_values_gt) - min(ty_values_gt))
#tz_filtered_nrmse = tz_filtered_rmse / (max(tz_values_gt) - min(tz_values_gt))
#yaw_filtered_nrmse = yaw_filtered_rmse / (max(yaw_values_gt) - min(yaw_values_gt))
#pitch_filtered_nrmse = pitch_filtered_rmse / (max(pitch_values_gt) - min(pitch_values_gt))
#roll_filtered_nrmse = roll_filtered_rmse / (max(roll_values_gt) - min(roll_values_gt))

# Calculate Filtered MAPE for each pose component
tx_filtered_mape = calculate_filtered_mape(tx_values_gt, tx_values_est)
ty_filtered_mape = calculate_filtered_mape(ty_values_gt, ty_values_est)
tz_filtered_mape = calculate_filtered_mape(tz_values_gt, tz_values_est)
yaw_filtered_mape = calculate_filtered_mape(yaw_values_gt, yaw_values_est)
pitch_filtered_mape = calculate_filtered_mape(pitch_values_gt, pitch_values_est)
roll_filtered_mape = calculate_filtered_mape(roll_values_gt, roll_values_est)

# Find Sample variance for tx, tz, and yaw
tx_sample_variance = np.var((np.array(tx_values_est) - np.array(tx_values_gt)), ddof=1)
tz_sample_variance = np.var((np.array(tz_values_est) - np.array(tz_values_gt)), ddof=1)
yaw_sample_variance = np.var((np.array(yaw_values_est) - np.array(yaw_values_gt)), ddof=1)

# Find sample STD for tx, tz, and yaw
tx_sample_std = np.std((np.array(tx_values_est) - np.array(tx_values_gt)), ddof=1)
tz_sample_std = np.std((np.array(tz_values_est) - np.array(tz_values_gt)), ddof=1)
yaw_sample_std = np.std((np.array(yaw_values_est) - np.array(yaw_values_gt)), ddof=1)

# Display Values
print(f"RMSE for Tx: {tx_rmse}, Ty: {ty_rmse}, Tz: {tz_rmse}, Yaw: {yaw_rmse}, Pitch: {pitch_rmse}, Roll: {roll_rmse}")
print("Filtered IQR RMSE for Tx:", tx_filtered_rmse, "Ty:", ty_filtered_rmse, "Tz:", tz_filtered_rmse, "Yaw:", yaw_filtered_rmse, "Pitch:", pitch_filtered_rmse, "Roll:", roll_filtered_rmse)
#print(f"Normalized RMSE over the range of ground truth values for Tx: {tx_nrmse}, Ty: {ty_nrmse}, Tz: {tz_nrmse}, Yaw: {yaw_nrmse}, Pitch: {pitch_nrmse}, Roll: {roll_nrmse}")
#print(f"Normalized filtered RMSE over range of ground truth values for Tx: {tx_filtered_nrmse}, Ty: {ty_filtered_nrmse}, Tz: {tz_filtered_nrmse}, Yaw: {yaw_filtered_nrmse}, Pitch: {pitch_filtered_nrmse}, Roll: {roll_filtered_nrmse}")
print("Median Absolute Error for Tx:", tx_medae, "Ty:", ty_medae, "Tz:", tz_medae, "Yaw:", yaw_medae, "Pitch:", pitch_medae, "Roll:", roll_medae)
print(f"Sample Variance for Tx: {tx_sample_variance}, Tz: {tz_sample_variance}, Yaw: {yaw_sample_variance}")
print(f"Sample Standard Deviation for Tx: {tx_sample_std}, Tz: {tz_sample_std}, Yaw: {yaw_sample_std}")