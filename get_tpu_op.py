import re

def extract_tpu_operators(filename):
    # 正则表达式匹配 "tpu." 后跟任意非空白字符的序列
    pattern = re.compile(r'tpu\.(\w+)')

    operators = set()

    # 打开并读取文件
    with open(filename, 'r') as file:
        for line in file:
            matches = pattern.findall(line)
            if matches:
                # 将找到的算子名称添加到集合中
                operators.update(matches)
    return list(operators)

def save_operators(operators, output_filename):
    # 将算子名称保存到文件
    with open(output_filename, 'w') as file:
        for operator in operators:
            file.write(operator + '\n')

filename = "/home/aigc/rzy_backup/realtimeASR/sherpa_sample/sherpa_work/joiner_work/sherpa_joiner_cv181x_f16_tpu.mlir"  # 源文件名
output_filename = 'joiner_tpu_operators.txt'  # 输出文件名

# 提取算子名称
operators = extract_tpu_operators(filename)

# 保存去重后的结果
save_operators(operators, output_filename)

print("TPU operators have been extracted and saved.")
