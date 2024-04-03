from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy import optimize
import csv
from similarities import Similarity


# 检查CUDA是否可用，然后选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载BERT模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained('bert-base-uncased').to(device)


# 定义计算相似度的函数
def calc_similarity(s1, s2):
    # 对句子进行分词，并添加特殊标记
    inputs = tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True)


    # 将inputs字典中的每个tensor移动到选定的设备上
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 将输入传递给BERT模型，并获取输出
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # print(embeddings)
    # 计算余弦相似度，并返回结果
    sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return sim
def calc_similarities(s1, s2):
    m = Similarity()
    r = m.similarity(s1, s2)
    return r.item()

def test():
    # 测试函数
    s1 = "文本相似度计算是自然语言处理中的一个重要问题"
    s2 = "自然语言处理中的一个重要问题是文本相似度计算"
    similarity = calc_similarity(s1, s2)
    print(f"相似度：{similarity:.4f}")


def read_csv(file_path):
    # 初始化数据
    data = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        last_index = -1
        for row in reader:
            cur_index = last_index + 1 if row['index'] == ''  else eval(row['index'])
            # print(cur_index, row['desc'])
            data[cur_index] = row['desc']
            last_index = cur_index
    return data

def format_matrix(cost_matrix):
    # 确定需要添加的虚拟列数
    n_rows, n_cols = cost_matrix.shape
    cost_matrix_square = cost_matrix
    if n_rows != n_cols:
        size_diff = abs(n_rows - n_cols)
        # 对于收益最大化问题：可以为缺少的行或列添加0或非常小的值，这样就不会对最终的分配方案产生不利影响。
        big_value = 0
        if n_rows > n_cols:
            # 添加虚拟列
            virtual_columns = big_value * np.ones((n_rows, size_diff))
            cost_matrix_square = np.hstack((cost_matrix, virtual_columns))
        else:
            # 添加虚拟行
            virtual_rows = big_value * np.ones((size_diff, n_cols))
            cost_matrix_square = np.vstack((cost_matrix, virtual_rows))
    return cost_matrix_square

def get_map_line(lines_src):
    map_line = []
    for line_src_index in lines_src:
        map_line.append(line_src_index)
    return map_line

def print_and_check_range_max(lines_src_1, lines_src_2, map_1, map_2, src):
    res = {}
    for i in range(len(src)):
        cur_max = 0
        cur_index = 0
        for j in range(len(src[i])):
            cur_max = max(cur_max, src[i][j])
            cur_index = j if cur_max == src[i][j] else cur_index 
        res[map_1[i]] = map_2[cur_index]
        print(lines_src_1[map_1[i]], lines_src_2[map_2[cur_index]], src[i][cur_index])
    return res

def print_and_check_range(row_ind, col_ind, lines_src_1, lines_src_2, map_1, map_2, cost):
    res = {}
    for i in range(len(row_ind)):
        # print(row_ind[i],col_ind[i])
        if row_ind[i] >= len(map_1) or col_ind[i] >= len(map_2):
            print("Error: index out of range")
            continue
        res[map_1[row_ind[i]]] = map_2[col_ind[i]]
        print(lines_src_1[map_1[row_ind[i]]], lines_src_2[map_2[col_ind[i]]], cost[row_ind[i]][col_ind[i]])
    return res

def main():
    # 读取数据
    lines_src_1 = read_csv('stopreason.csv')
    lines_src_2 = read_csv('stopreason_cd.csv')
    # 初始化映射
    map_1 = get_map_line(lines_src_1)
    map_2 = get_map_line(lines_src_2)
    # 遍历数据
    res = []
    for line_src_1_index, line_src_1_value in lines_src_1.items():
        res_line = []
        for line_src_2_index, line_src_2_value in lines_src_2.items():
            # 计算相似度
            # similarity = calc_similarity(line_src_1_value, line_src_2_value)
            similarity = calc_similarities(line_src_1_value, line_src_2_value)
            print(f"句子1：{line_src_1_value}")
            print(f"句子2：{line_src_2_value}")
            print(f"相似度：{similarity:.4f}")
            res_line.append(similarity)
            print()
        res.append(res_line)
    print(res)    

    print("取最相似")
    res1 = print_and_check_range_max(lines_src_1, lines_src_2, map_1, map_2, res)
    print(res1)

    print("匈牙利算法")
    cost = format_matrix(np.array(res))
    row_ind,col_ind=optimize.linear_sum_assignment(cost)
    res2 = print_and_check_range(row_ind, col_ind, lines_src_1, lines_src_2, map_1, map_2, cost)
    print(res2)
    print(cost[row_ind,col_ind].sum())#数组求和　　输出：[0 1 2][1 0 2] [1 2 2] 5

if __name__ == '__main__':
    main()