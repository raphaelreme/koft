set -e

# Fake@100%
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.0 --detection.fake.fnr 0.0

# Fake@90%
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1

# Fake@80%
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.2 --detection.fake.fnr 0.2

# Fake@70%
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3

# Fake@60%
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4
$RUN_KOFT_ENV expyrun config/track/dupre.yml --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.4 --detection.fake.fnr 0.4
