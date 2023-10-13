set -e

# Springs - 100 particles
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 100 --simulator.imaging_config.snr 1 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 100 --simulator.imaging_config.snr 2 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 100 --simulator.imaging_config.snr 4 --seed $@


# Springs - 400 particles
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 400 --simulator.imaging_config.snr 1 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 400 --simulator.imaging_config.snr 2 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 400 --simulator.imaging_config.snr 4 --seed $@


# Springs - 1000 particles
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 1 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 2 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/springs.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 4 --seed $@


# Flow - 100 particles
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 100 --simulator.imaging_config.snr 1 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 100 --simulator.imaging_config.snr 2 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 100 --simulator.imaging_config.snr 4 --seed $@


# Flow - 400 particles
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 400 --simulator.imaging_config.snr 1 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 400 --simulator.imaging_config.snr 2 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 400 --simulator.imaging_config.snr 4 --seed $@


# Flow - 1000 particles
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 1 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 2 --seed $@
conda run -n visual_tracking --live-stream expyrun config/simulator/flow_lovas.yml --simulator.particle.n 1000 --simulator.imaging_config.snr 4 --seed $@