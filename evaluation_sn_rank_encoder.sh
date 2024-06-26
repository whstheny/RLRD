PROJECT_DIR=/mnt/nfs-storage-pvc-n28/user_codes/rizeJin/wxt/RankCSE-master

SEED=61507
python evaluation_rank.py \
    --model_name_or_path $PROJECT_DIR/runs/checkpoint-3-20/ \
    --task_set sts \
    --mode test
