import jieba
import re
import numpy as np
import os

# 是一个二维列表, vocabuary[i][x, y]表示第i个单词, 在健康邮件中出现的次数(或者说出现有该单词的健康邮件的数目)为x, 在垃圾邮件中出现的次数是y
vocabulary = []     # 维持一个词汇表, 存储训练语料库里所有的单词, 以及它们分别在healthy email和spam的出现次数
word_to_id_map = {}  # 构造一个字典，存放vocabulary中单词和位置的映射
quantity_email = [0, 0]             # 一维列表, q[0]是训练集中健康邮件的数量, q[1]是训练集中垃圾邮件的数量
# 获取停用词列表
def get_stop_words():
	stop_words = []
	with open('../data/中文停用词表.txt', 'r', encoding='gbk') as reader:
		for word in reader.readlines():
			stop_words.append(word.strip())
	return stop_words

# 对输入的文档, 删除掉停用词
def remove_stop_words(pre_list):
	after_result = []
	stop_words = get_stop_words()  # 停用词列表;
	for w in pre_list:
		if w not in stop_words:
			after_result.append(w)
	# print(after_result)
	return after_result


# 读取指定文件夹下所有文件, 进行训练过程
# path = '../data/normal' or 'spam'
# sign = 0 or 1, 0对应的是健康邮件, 1代表spam
def read_file_to_train(path, sign):
	files = os.listdir(path)
	# 记录两类邮件数
	quantity_email[sign] = len(files)

	for name in files:
		file = os.path.join(path, name)     # 单个文件的完整路径
		calculate_occurrence_count(file, sign)

# 该篇邮件中所有出现的单词, 在该分类下的出现次数都加1
def calculate_occurrence_count(file, sign):

	with open(file, 'r', encoding='gbk') as reader:
		# 过滤掉非中文字符
		rule = re.compile(r"[^\u4e00-\u9fa5]")
		line = reader.read()
		content = rule.sub('', line)
		initial_words = jieba.lcut(content)  # jieba分词获得初始文字列表
		processed_words = remove_stop_words(initial_words)  # 删除停用词

	words = list(set(processed_words))        # 去掉列表中的重复值
	for w in words:
		if w not in word_to_id_map.keys():
			vocabulary.append([0, 0])               # 在词汇表中新增一个位置记录两个频数
			word_to_id_map[w] = len(vocabulary)-1   # 在映射map里记录好单词w和id的对应关系
		vocabulary[word_to_id_map[w]][sign] +=1                    # 当前单词在sign分类下频数+1
	# print(vocabulary)
	# print(word_to_id_map)

# 结合输出vocabulary和word_to_id_map
def try_show():
	for w in word_to_id_map.keys():
		id = word_to_id_map[w]
		print(w + '\t' + str(vocabulary[id][0]) + '\t' + str(vocabulary[id][1]))

# 将频数转换成频率
def calculate_occurrence_frequency():
	for i in range(0, len(vocabulary)):
		vocabulary[i][0] = vocabulary[i][0] / quantity_email[0] #每个位置的单词次数除以总的邮件数
		vocabulary[i][1] = vocabulary[i][1] / quantity_email[1]

		if vocabulary[i][0] == 0.0:     # 如果一个词在健康邮件中出现次数为0, 为了避免概率为0, 影响连乘计算, 通过加普拉斯平滑系数修正
			vocabulary[i][0] = (vocabulary[i][0] +1)/ (quantity_email[0]+len(vocabulary))
		if vocabulary[i][1] == 0.0:
			vocabulary[i][1] = (vocabulary[i][1] +1)/ (quantity_email[1]+len(vocabulary))

# 将模型的参数保存到本地, 实现持久化
# vocabulary, word_to_id_map
def write_model_to_file():
	file_vocabulary = '../data/file_vocabulary.txt'
	with open(file_vocabulary, 'w') as writer:          # open, 若不存在, 自动会新建
		for i in range(0, len(vocabulary)):
			writer.write(str(vocabulary[i][0]) + '\t' + str(vocabulary[i][1]) + '\n')

	file_id_map = '../data/file_id_map.txt'

	with open(file_id_map, 'w') as writer:
		for key, value in word_to_id_map.items():
			writer.write(str(key) + '\t' + str(value) + '\n')

if __name__ == "__main__":

	read_file_to_train('../data/normal', 0)
	read_file_to_train('../data/spam', 1)
	calculate_occurrence_frequency()
	write_model_to_file()

