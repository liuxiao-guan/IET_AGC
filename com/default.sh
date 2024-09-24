#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

log_dir="/root/autodl-tmp/logs/Projects/IET-AGC/cifar10_default/"
config_file="/root/autodl-fs/Project/IET-AGC/config/CIFAR10.txt"
mem_ratio=2.0
#python /root/autodl-fs/Project/IET-AGC/fedavg_es/main_fed.py --logdir="$log_dir" --mem_ratio="$mem_ratio" --parallel --flagfile="$config_file" 
python /root/autodl-fs/Project/IET-AGC/main.py --logdir="$log_dir"  --parallel --flagfile="$config_file" 

my_list=("ckpt399k.pt")

for item in "${my_list[@]}"
do 
    unet_dir="${log_dir}/${item}"
    echo "正在分析的模型是：$unet_dir"

    last_number=$(echo "$unet_dir" | awk -F'[^0-9]+' '{print $(NF-1)}')

    # 指定执行次数
    repeat_count=2

    # 循环执行命令
    for ((i=1; i<=$repeat_count; i++))
    do
        generate_dir="${log_dir}/${last_number}_nema/${i}/generate"
        similar_dir="${log_dir}/${last_number}_nema/${i}/similar"
        grid_dir="${log_dir}/${last_number}_nema/${i}/grid"

        # 在这里替换成你要执行的命令，$i 表示当前的循环次数
        python /root/autodl-fs/Project/IET-AGC/attack/all_in.py --flagfile="$config_file" --generate_dir="$generate_dir" --similar_dir="$similar_dir" --grid_dir="$grid_dir" --unet_dir="$unet_dir" --parallel --batch_size=128 --use_ema=False -Fed=False 

    done
done 



