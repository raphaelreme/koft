set -e

# Run for different f1/psnr/delta/seed with warped (CP/CV) SKT, (CP/CV) trackmate and eMHT
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --tracking_method skt --warp True --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --kalman.order 0 --__run__.__output_dir__ experiment_folder/tracking_warp
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --tracking_method skt --warp True --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --kalman.order 1 --__run__.__output_dir__ experiment_folder/tracking_warp
# $RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6--tracking_method skt --warp True --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --kalman.order 2 --__run__.__output_dir__ experiment_folder/tracking_warp
# Uncomment to also run trackmate (Trackmate does not work well (either kf linear but not suited, or no kf at all))
# $RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --tracking_method trackmate --warp True --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --kalman.order 0 --__run__.__output_dir__ experiment_folder/tracking_warp
# $RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --tracking_method trackmate-kf --warp True --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $4 --kalman.order 1 --__run__.__output_dir__ experiment_folder/tracking_warp
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --tracking_method emht --warp True --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --__run__.__output_dir__ experiment_folder/tracking_warp

# Find best KOFT (CP/CA)
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --kalman.order 1 --__run__.__output_dir__ experiment_folder/tracking_warp
$RUN_KOFT_ENV expyrun config/track/snr.yml --simulation_name $1 --alpha $2 --delta $3 --seed $6 --detection.detector $4 --detection.fake.fpr $5 --detection.fake.fnr $5 --kalman.order 2 --__run__.__output_dir__ experiment_folder/tracking_warp
