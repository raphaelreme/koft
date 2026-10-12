set -e

# Run long term tracking

# Springs
$RUN_KOFT_ENV expyrun config/track/long_term.yml --tracking_methods skt
$RUN_KOFT_ENV expyrun config/track/long_term.yml --tracking_methods koft
$RUN_KOFT_ENV expyrun config/track/long_term.yml --tracking_methods trackmate-kf --max_length 2000
$RUN_KOFT_ENV expyrun config/track/long_term.yml --tracking_methods emht --max_length 2000
$RUN_KOFT_ENV expyrun config/track/long_term.yml --tracking_methods trackastra-lap --max_length 2000
$RUN_KOFT_ENV expyrun config/track/long_term.yml --tracking_methods trackastra --max_length 500
