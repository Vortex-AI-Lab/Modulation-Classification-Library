export CUDA_VISIBLE_DEVICES=0

model=MCformer
dataset=RML2016a

python main.py \
  --model $model \
  --dataset $dataset \
  --file_path dataset/RML2016.10a_dict.pkl \
  --batch_size 16 \
  --num_epochs 50 \
  --learning_rate 0.0001 \
  --optimizer adam \
  --criterion cross_entropy \
  --patience 5 \
