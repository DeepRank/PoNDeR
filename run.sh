wget -P $1 $3
python train.py --data_path $1 --out_folder $2  --num_epoch 3 --cosine_decay --batch_size 100 --num_workers 4 --num_points 500| tee $2/log.txt