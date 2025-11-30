# Define
export PYTHONPATH=/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/maps"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"

CONFIG_NAME=default

# training
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=WoTE/${CONFIG_NAME} \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=30  \
split=trainval 

# evaluation
python ./navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp/WoTE/default/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest \
