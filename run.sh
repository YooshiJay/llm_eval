#!/bin/bash

# 检查是否提供了参数
if [ "$#" -ne 1 ]; then
    echo "用法: $0 <要执行的 .ipynb 文件名>"
    exit 1
fi

dir=$(pwd)

# 获取传入的参数
ipynb_file="$dir/$1"

# 检查输入文件是否存在
if [[ ! -f "$ipynb_file" ]]; then
    echo "文件不存在: $ipynb_file"
    exit 1
fi

# 获取文件名（不包括扩展名）
filename="${ipynb_file%.*}"

# 将 .ipynb 文件转换为 .py 文件
jupyter nbconvert --to script "$ipynb_file"

# 检查转换是否成功
py_file="${filename}.py"
if [[ ! -f "$py_file" ]]; then
    echo "转换失败: $ipynb_file"
    exit 1
fi

# 在后台运行 .py 文件
nohup python "$py_file" &

# 输出提示信息
echo "已在后台运行: $py_file"
