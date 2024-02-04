set -e

# Run for springs and flow with Fake detection with F1 at 80% and PSNR from 0.1 to 10.0 and Delta 5.0 to 500.0

# Springs
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 0.1 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 0.1 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 0.1 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 0.5 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 0.5 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 0.5 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 1.5 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 1.5 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 1.5 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 3.0 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 3.0 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 3.0 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 10.0 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 10.0 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion springs --psnr 10.0 --delta 500.0 --seed $@


# Flow
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 0.1 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 0.1 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 0.1 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 0.5 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 0.5 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 0.5 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 1.5 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 1.5 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 1.5 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 3.0 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 3.0 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 3.0 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 10.0 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 10.0 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/track/snr.yml --motion flow_20140829_1 --psnr 10.0 --delta 500.0 --seed $@
