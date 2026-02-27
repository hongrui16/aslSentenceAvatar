#!/bin/bash



# 设置目标目录（可传参，默认当前目录）
TARGET_DIR="${1:-.}"

echo "开始更新 $TARGET_DIR 下所有文件的时间戳..."

# 更新所有文件的 mtime 和 atime 为当前时间
find "$TARGET_DIR" -type f -exec touch {} +

# 间接更新 ctime：对每个文件执行 chmod "保持不变"，从而触发 ctime 变化
find "$TARGET_DIR" -type f -exec chmod --reference={} {} \;

echo "✅ 所有文件的 atime、mtime 已更新，ctime 也已间接更新：$(date)"
