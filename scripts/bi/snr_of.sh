set -e

# Run of for all kinds of snr videos

$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 0.1 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 0.1 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 0.1 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 0.5 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 0.5 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 0.5 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 1.5 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 1.5 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 1.5 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 3.0 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 3.0 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 3.0 --delta 500.0 --seed $@


$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 10.0 --delta 5.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 10.0 --delta 50.0 --seed $@
$RUN_KOFT_ENV expyrun config/optical_flow/snr.yml --psnr 10.0 --delta 500.0 --seed $@
