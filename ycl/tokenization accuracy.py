# -*- coding: utf-8 -*-
# @Author   :Yangyyz
import pkuseg
import jieba
import thulac
import re
import pip

'''
def get_prf(listokenization, listhuman):
    e = 0
    c = 0
    N = len(listhuman)
    for x in listokenization:
        if x in listhuman:
            c = c + 1

    e = len(listokenization) - c
    P = c / (c + e)
    R = c / N
    F = (2 * P * R) / (P + R)
    FN=N-c
    TN=229-e
    print(len(listokenization))
    print("P:", P, "\nR:", R, "\nF:", F)

    print('TP:', c)
    print('FP:', e)
    print('FN:', FN)
    print('TN:', TN)
'''

f = open("express.txt",encoding='UTF-8')#读取原始摘要
s = f.read()
# print(s)

# 标准分词
f1 = open("standard_express.txt",encoding='UTF-8')
s1 = ['由于', 'ZrO2', '-', 'C质', '耐火材料', '具有', '优异', '的', '抗', '侵蚀性', '，', '因此', '它', '被', '广泛', '应用', '于', '连铸用', '浸入式', '水口', '渣线', '部位', '以及', '塞棒', '的', '棒头', '部位', '。', '本文', '介绍', '了', 'ZrO2', '-', 'C质', '耐火材料', '的', '主要', '原料', '，', '阐述', '了', 'ZrO2', '-', 'C质', '耐火材料', '的', '蚀损', '机理', '以及', '改进', '抗', '侵蚀性','的', '措施', '。','材料', '化学', '是', '一门', '新兴', '的', '交叉', '学科', ',', '其', '专业', '实践', '课程', '是', '培养', '好', '专门', '人才', '的', '一个', '重要', '途径', '.', '本文', '以', '工科', '应用型', '人才', '培养', '为', '目标', '背景', ',', '根据', '我校', '材料', '化学', '专业', '十一年', '来', '的', '实习', '实践', '教学', '经验', ',', '提出', '了', '当前', '实习', '实践', '教学', '存在', '的', '一些', '普遍性', '问题', ',', '并', '对', '其', '展开', '了', '分析', '与', '讨论', '.', '提出', '构建', '完整', '的', '实践', '教学', '体系', ',', '学校', '应', '建立', '稳固', '的', '实习', '实践', '基地', ',', '加强', '实习', '教学', '管理', '、', '教学', '改革', '以及', '加强', '对', '学生', '实践', '能力', '的', '培养', '.','以', 'SR', '、', '纳米', 'Fe3O4', '和', '纳米', 'MH', '为', '主要原料', '制备', 'MH／Fe3O4／SR', '磁性', '橡胶', '复合材料', '。', '研究', '纳米', 'Fe3O4', '和', '纳米MH', '不同', '配比', '时', '，', '复合材料', '的', '物理力学', '性能', '变化', '、', '耐热', '以及', '摩擦', '性能', '  变化', '。', '结果', '表明', '：', '纳米粒子', '在', 'SR', '基体', '中', '分布', '较为', '均匀', '，', '不同', '配比', '的', 'Fe3O4／MH', '能够', '有效', '改善', '硅', '橡胶', '的', '物理力学', '性能', '。', '当', '配比', '20phrMH／10', 'phrFe3O4', '时', '，', '复合', '的', '拉伸', '强度', '、', '伸长率', '有所', '改善']
# print(s1)

#l_std = re.split(r'/', s1)
N = len(s1)
l_jieba = jieba.lcut(s, cut_all=False)
print("标准分割: ", s1)
print('jieba',l_jieba)
print("标准分割的单词数N：", N)

print(len(s1))
print(len(l_jieba))
for i in range(len(s1)):
    #print(s1[i])
    a = len(s1[i])
    if s1[i] is not '\n':
      print('0'*(a-1),'1',end=' ')



# jieba分词

# print("jieba分词：", l_jieba)
# get_prf(l_jieba, s1)
print('****')
for i in range(len(l_jieba)):
    a = len(l_jieba[i])
    if l_jieba[i] is not '\n':
        print('0'*(a-1),'1',end=' ')
'''
# thulac分词
thu = thulac.thulac(seg_only=True)
s_thu = thu.cut(s)
# print(s_thu)

l_thu = []
for x in s_thu:
    l_thu.append(x[0])
print("thulac分词：", l_thu)
get_prf(l_thu, s1)

seg = pkuseg.pkuseg()
text = seg.cut(s)  # 进行分词
print(text)
get_prf(text, s1)
'''