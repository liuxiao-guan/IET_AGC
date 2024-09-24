#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

log_dir="/root/autodl-tmp/logs/Projects/IET-AGC/cifar10_c10/"
config_file="/root/autodl-fs/Project/IET-AGC/config/CIFAR10.txt"
mem_ratio=2.0
# 使用正则表达式提取数字部分，并赋值给变量
# if [[ $log_dir =~ es_([0-9.]+) ]]; then
#     mem_ratio=${BASH_REMATCH[1]}
# else
#     echo "无法从地址中提取数字。"
# fi

#python /root/autodl-fs/Project/IET-AGC/fedavg_es/main_fed.py --logdir="$log_dir" --mem_ratio="$mem_ratio" --parallel --flagfile="$config_file" 
#python /root/autodl-fs/Project/IET-AGC/main.py --logdir="$log_dir"  --parallel --flagfile="$config_file" 

my_list=("global_ckpt_round10.pt")

for item in "${my_list[@]}"
do 
    unet_dir="${log_dir}/${item}"
    echo "正在分析的模型是：$unet_dir"

    last_number=$(echo "$unet_dir" | awk -F'[^0-9]+' '{print $(NF-1)}')

    # 指定执行次数
    repeat_count=3

    # 循环执行命令
    for ((i=2; i<=$repeat_count; i++))
    do
        generate_dir="${log_dir}/${last_number}_nema/${i}/generate"
        similar_dir="${log_dir}/${last_number}_nema/${i}/similar"
        grid_dir="${log_dir}/${last_number}_nema/${i}/grid"

        # 在这里替换成你要执行的命令，$i 表示当前的循环次数
        python /root/autodl-fs/Project/IET-AGC/attack/all_in.py --flagfile="$config_file" --generate_dir="$generate_dir" --similar_dir="$similar_dir" --grid_dir="$grid_dir" --unet_dir="$unet_dir" --parallel --batch_size=128 --use_ema=False 

    done
done 
# unet_dir="${log_dir}/global_ckpt_round10.pt"

# last_number=$(echo "$unet_dir" | awk -F'[^0-9]+' '{print $(NF-1)}')

# # 指定执行次数
# repeat_count=1


# # 循环执行命令
# for ((i=1; i<=$repeat_count; i++))
# do
#     generate_dir="${log_dir}/${last_number}_nema/${i}/generate"
#     similar_dir="${log_dir}/${last_number}_nema/${i}/similar"
#     grid_dir="${log_dir}/${last_number}_nema/${i}/grid"

#     # 在这里替换成你要执行的命令，$i 表示当前的循环次数
#     python /home/lx/IET-AGC-main/attack/all_in.py --flagfile="$config_file" --generate_dir="$generate_dir" --similar_dir="$similar_dir" --grid_dir="$grid_dir" --unet_dir="$unet_dir" --parallel --batch_size=256 --use_ema=False

# done


