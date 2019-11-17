import os
import src.train as train
import re
import jieba
import copy

# 垃圾邮件和健康邮件的先验概率
p_s = 0.5
p_h = 0.5
conditional_hw_and_sw = []
conditional_wh_and_ws = []

word_to_id_map = {}

# 计算一篇邮件中的P(s|W)和P(h|W), 并返回;
def calculate_conditional_sW(file):
	with open(file, 'r', encoding='gbk') as reader:
		# 过滤掉非中文字符
		rule = re.compile(r"[^\u4e00-\u9fa5]")
		line = reader.read()
		content = rule.sub('', line)
		initial_words = jieba.lcut(content)  # jieba分词获得初始文字列表
		processed_words = train.remove_stop_words(initial_words)  # 删除停用词
	words = list(set(processed_words))      # 去掉列表中的重复值
	p_sW = 1        # 代表的是P(s|w_1, w_2, .......,w_n)
	p_Ws = 1        # 代表, P(w_1, w_2,......, w_n | s)
	p_Wh = 1
	for i in range(0, len(words)):
		# id的转换
		if words[i] in word_to_id_map.keys():
			id = word_to_id_map[words[i]]
			p_Ws *= conditional_wh_and_ws[id][1]  # 朴素贝叶斯模型, 假设变量的各特征是相互独立的
			p_Wh *= conditional_wh_and_ws[id][0]  # 所以, P(w_1,w_2,.....,w_n|s) = P(w_1|s)*P(w_2|s)*....*P(w_n|s)
		else:
			p_Ws *= 0.4         # 如果一个单词之前从没出现过, 无法从历史资料中获取P(w|s), 假定其等于0.4,
			p_Wh *= 0.6         # 因为垃圾邮件用的往往是固定的词语, 如果这个单词从没出现过, 那它多半是正常的词

	# p_sW = (p_Ws * p_s) / ()
	# p_hW = (p_Wh * p_h) / ()
	p_sW = (p_Ws * p_s)             # 因为分母一样, 所以只考虑分子最大化
	p_hW = (p_Wh * p_h)
	return p_sW, p_hW

# 从指定文件夹下读取全部文件,进行条件概率P(s|W)的运算
def read_files_to_predict(path):
	files = os.listdir(path)
	category = {}               # 记录分类
	for name in files:
		file = os.path.join(path, name)
		(p_sW, p_hW) = calculate_conditional_sW(file)
		if p_sW > p_hW:
			category[name] = 1      # 1表示垃圾邮件
		else:
			category[name] = 0      # 0表示健康邮件
	return category


# 计算并输出准确率
def show_result(category):
	accuracy = 1                # 准确率
	correct_quantity = 0        # 被正确分类的数目
	for name in category.keys():
		if int(name) >= 200:        # 则应该是垃圾邮件
			if category[name] == 1:
				correct_quantity += 1
			print(name,"实际为: 1","预测为：",category[name])
		else:                       # 应该是健康邮件
			if category[name] == 0:
				correct_quantity += 1
			print(name, "实际为: 0","预测为：", category[name])
	accuracy = correct_quantity / len(category.keys())

	print("精确率为: "+ str(accuracy*100)+"%")

# 从持久化的文件中读取模型的参数, 填充给相应变量
def read_model_from_file():
	file_vocabulary = '../data/file_vocabulary.txt'
	file_id_map = '../data/file_id_map.txt'

	with open(file_vocabulary, 'r') as reader:
		for line in reader.readlines():
			list = line.strip().split('\t')
			if len(list) == 2:
				conditional_wh_and_ws.append([float(list[0]), float(list[1])])
	with open(file_id_map, 'r') as reader:
		for line in reader.readlines():
			list = line.strip().split('\t')
			if len(list) == 2:
				word_to_id_map[list[0]] = int(list[1])

if __name__ == '__main__':
	# 词集模式, 只关心某个单词是否出现在邮件中, 至于出现次数>1, 并不关心;
	# 词袋模式, 不仅关心单词是否出现, 还关心单词出现的次数;
	#加载模型文件
	read_model_from_file()
	#读取文件并进行预测每个邮件的类别（通过计算每封邮件的p_sW和p_hW）
	category = read_files_to_predict('../data/test')
	#显示结果
	show_result(category)