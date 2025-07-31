#!/bin/bash
# evil --trace-fork-before-exec
# PIPE_ON=1 nsys profile \
#         --trace=cuda,nvtx --output=moe_profile_nvtx_test \
#         --force-overwrite=true \
#         --trace-fork-before-exec=true \
#         python prof_dp.py \
#         --model="../scripts/DeepSeek-V3" \
#         --dp-size=2 \
#         --tp-size=2 \
#         --input-file="/userhome/cs3/zhoujoey/foldinfer/workspace/4k_token.txt"

#evil --trace-fork-before-exec
# PIPE_ON=1 nsys profile \
#         --trace=cuda,nvtx --output=moe_profile_nvtx_test \
#         --force-overwrite=true \
#         --trace-fork-before-exec=true \
PIPE_ON=1 python prof_dp.py \
--model="../scripts/DeepSeek-V3" \
--dp-size=1 \
--tp-size=2 \
--input-file="/userhome/cs3/zhoujoey/foldinfer/workspace/8k_token.txt"

# times=()

# # 运行10次
# for i in {1..20}; do
#     echo "第 $i 次运行..."
    
#     # 执行命令并捕获结果
#     raw_results=$(PIPE_ON=1 python prof_dp.py \
#         --model="../scripts/DeepSeek-V3" \
#         --dp-size=2 \
#         --tp-size=2 \
#         --input-file="/userhome/cs3/zhoujoey/foldinfer/workspace/32k_token.txt" | \
#         grep "forward time:" | awk '{print $(NF-1)}')
    
#     if [ -n "$raw_results" ]; then
#         # 将多行结果转换为数组
#         readarray -t round_values <<< "$raw_results"
        
#         # 显示本轮的所有值
#         echo "  本轮四个值: ${round_values[@]}"
        
#         # 计算本轮平均值（去除最大最小值）
#         round_avg=$(printf "%s\n" "${round_values[@]}" | awk '
#         {
#             values[NR] = $1
#             sum += $1
#             count++
#             if (min == "" || $1 < min) min = $1
#             if (max == "" || $1 > max) max = $1
#         }
#         END {
#             if (count > 2) {
#                 # 去除最大最小值后计算平均值
#                 trimmed_sum = sum - min - max
#                 trimmed_count = count - 2
#                 printf "%.6f", trimmed_sum/trimmed_count
#             } else if (count > 0) {
#                 # 如果数据不足4个，使用全部数据
#                 printf "%.6f", sum/count
#             }
#         }')
        
#         if [ -n "$round_avg" ]; then
#             times+=($round_avg)
#             echo "  本轮去除最大最小值后平均值: $round_avg"
#         else
#             echo "  警告: 第 $i 次运行无法计算平均值"
#         fi
#     else
#         echo "  警告: 第 $i 次运行未获得结果"
#     fi
    
#     # 添加短暂延迟避免系统负载过高
#     sleep 1
# done

# echo "================================"
# echo "所有运行完成！"
# echo ""
# echo "结果汇总:"
# echo "--------------------------------"

# # 打印所有轮次的平均值
# for i in "${!times[@]}"; do
#     printf "第 %2d 轮平均值(去最大最小): %s\n" $((i+1)) "${times[i]}"
# done

# # 如果有足够的结果，计算统计信息
# if [ ${#times[@]} -gt 0 ]; then
#     echo ""
#     echo "统计信息:"
#     echo "--------------------------------"
#     echo "成功运行轮次: ${#times[@]}"
    
#     # 计算十轮平均值的统计信息，同时再次去除最大最小值
#     printf "%s\n" "${times[@]}" | awk '
#     BEGIN { 
#         sum=0; min=""; max=""; count=0 
#         # 存储所有值用于去除最大最小值
#     }
#     {
#         values[NR] = $1
#         sum += $1
#         count++
#         if (min == "" || $1 < min) min = $1
#         if (max == "" || $1 > max) max = $1
#     }
#     END {
#         if (count > 0) {
#             printf "十轮总平均值(含全部数据): %.6f\n", sum/count
#             printf "各轮平均值最小: %s\n", min
#             printf "各轮平均值最大: %s\n", max
            
#             # 如果有足够数据，计算去除最大最小值后的平均值
#             if (count > 2) {
#                 trimmed_sum = sum - min - max
#                 trimmed_count = count - 2
#                 printf "十轮去除最大最小值后平均值: %.6f\n", trimmed_sum/trimmed_count
#                 printf "使用数据量: %d/%d 轮\n", trimmed_count, count
#             } else {
#                 printf "数据不足，无法去除最大最小值\n"
#             }
            
#             printf "数据范围: %s - %s\n", min, max
#         }
#     }'
# fi