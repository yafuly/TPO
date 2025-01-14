SERVER_MODEL="server-allenai/Llama-3.1-Tulu-3-70B-SFT"
IP=41
PORT=8000
max_tokens_all=8192

DATA_DIR="/mnt/petrelfs/liyafu/Work/Github/TPO/data"
MODEL=`basename $SERVER_MODEL`


SEED=7
data="sample.json"
echo $data
DATA_PATH=$DATA_DIR/$data


tpo_mode="tpo"
sample_size=5
srun python run.py --ip $IP --data_path $DATA_PATH --server_model $SERVER_MODEL --port $PORT --tpo_mode $tpo_mode --max_tokens_response 2048 --max_tokens_all $max_tokens_all --sample_size $sample_size --seed $SEED --max_iterations 2 --num_threads 4

