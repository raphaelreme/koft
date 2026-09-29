set -e

$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4
