set -e

# Springs
# Fake@70%
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3

# Fake@90%
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1

# Wavelet
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/111/exp.0/ --seed 111 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/222/exp.0/ --seed 222 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/333/exp.0/ --seed 333 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/444/exp.0/ --seed 444 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name springs/1000/1.5-50.0/555/exp.0/ --seed 555 --tracking_method $@


#Flow
# Fake@70%
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3

# Fake@90%
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1

# Wavelet
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/111/exp.0/ --seed 111 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/222/exp.0/ --seed 222 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/333/exp.0/ --seed 333 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/444/exp.0/ --seed 444 --tracking_method $@
$RUN_KOFT_ENV expyrun config/track/simulator.yml --simulation_name flow_20140829_1/1000/1.5-50.0/555/exp.0/ --seed 555 --tracking_method $@
