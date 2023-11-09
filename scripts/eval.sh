set -e

# Springs
# Fake@70%
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3

# Fake@90%
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1

# Wavelet
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/111/exp.0/ --seed 111 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/222/exp.0/ --seed 222 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/333/exp.0/ --seed 333 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/444/exp.0/ --seed 444 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name springs/1000/1.8/555/exp.0/ --seed 555 --tracking_method $@


#Flow
# Fake@70%
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.3 --detection.fake.fnr 0.3

# Fake@90%
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/111/exp.0/ --seed 111 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/222/exp.0/ --seed 222 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/333/exp.0/ --seed 333 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/444/exp.0/ --seed 444 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/555/exp.0/ --seed 555 --tracking_method $@ --detection.detector fake --detection.fake.fpr 0.1 --detection.fake.fnr 0.1

# Wavelet
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/111/exp.0/ --seed 111 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/222/exp.0/ --seed 222 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/333/exp.0/ --seed 333 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/444/exp.0/ --seed 444 --tracking_method $@
conda run -n visual_tracking --live-stream expyrun config/track/default.yaml --simulation_name flow_20140829_1/1000/1.8/555/exp.0/ --seed 555 --tracking_method $@
