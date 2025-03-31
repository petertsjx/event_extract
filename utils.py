import pandas as pd
import re
import jieba
import numpy as np
from collections import Counter

from numba.np.arrayobj import normalize_axis


def deduplicate_with_duplicates(input_list):
    """
    方法1: 基础实现
    去重并返回重复元素

    参数:
    input_list: 原始列表

    返回:
    去重后的列表和重复元素列表
    """
    # 去重后的列表
    unique_list = list(dict.fromkeys(input_list))

    # 使用 Counter 统计元素出现次数
    count_dict = Counter(input_list)

    # 找出重复元素
    duplicate_elements = [
        item for item, count in count_dict.items()
        if count > 1
    ]

    return {
        'unique_list': unique_list,
        'duplicate_elements': duplicate_elements
    }
def clean_chinese_text(text):
    """
    清洗中文新闻文本，去除符号和英文字符

    参数:
        text: 输入的原始文本
    返回:
        清洗后的纯中文文本
    """
    # 移除所有非中文字符（保留中文、数字和空格）
    # \u4e00-\u9fff 是中文字符的Unicode范围
    text = str(text)
    cleaned_text = re.sub(r'[^\u4e00-\u9fff\s0-9]', '', text)

    # 合并多个空格为一个
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    #去掉数字
    cleaned_text = re.sub(r'[0-9]+', '', cleaned_text)
    return cleaned_text.strip()

def count_n_ns(sentence):
    count=0
    kw = jieba.analyse.extract_tags(sentence, topK=10, withWeight=True, allowPOS=('n', 'ns'))
    return len(kw)

def process_full_test_data(path):
    df = pd.read_csv(path)
    df['title']=df['title'].apply(clean_chinese_text)
    df['description'] = df['description'].apply(clean_chinese_text)
    return df[['id','title','description']]


def read_tag(tag, text):
    """
    提取指定标签中的内容

    Args:
        tag: 开始标签，如 "<title>"
        text: 包含标签的文本

    Returns:
        匹配到的标签内容列表
    """
    # 构造结束标签
    tag_start = re.escape(tag)
    tag_end = re.escape(tag.replace('<', '</'))

    # 正则表达式模式
    pattern = f"{tag_start}(.*?){tag_end}"

    # 使用非贪婪匹配
    result = re.findall(pattern, text, re.DOTALL)
    return result

def add_before_save(res):
    event=read_tag("<event>",res)
    title=read_tag("<title>",res)
    action = read_tag("<action>", res)
    entity = read_tag("<entity>", res)
    id = read_tag("<id>", res)
    if event==[]:
        e=title[0]
    elif "的结果" in event[0]:
        e=title[0]
    else:
        e=event[0]
    if action==[]:
        a=title[0]
    elif "的结果" in action[0]:
        a=title[0]
    else:
        a=action[0]

    if entity==[]:
        en=title[0]
    elif "的结果" in entity[0]:
        en=title[0]
    else:
        en=entity[0]
    normalize_string="<id>"+id[0]+"</id>"+"<title>"+title[0]+"</title>"+"<event>"+e+"</event>"+"<action>"+a+"</action>"+"<entity>"+en+"</entity>"
    if "<bad>" in res:
        return normalize_string+"<bad>"+"\n"
    else:
        return normalize_string+"\n"

def read_data_for_chroma(path,tag):
    results = []
    ids = []
    with open(path, 'r', encoding='utf-8') as fp:
        ll = fp.readlines()
        for l in ll:
            if "<bad>" in l:
                #print("bad data, continue...")
                continue
            tag_content = read_tag(tag,l)
            id = read_tag("<id>",l)
            if tag_content!=[]:
                results.append(tag_content[0])
            else:
                results.append("")
            if [] not in id:
                ids.append(id[0])
            else:
                ids.append("")
    return results,ids

def compute_cosine_similarity(
                              embedding1: np.ndarray,
                              embedding2: np.ndarray) -> float:
    """
    计算两个 embedding 之间的余弦相似度

    参数:
    - embedding1: 第一个 embedding 向量
    - embedding2: 第二个 embedding 向量

    返回:
    - 余弦相似度值（-1 到 1 之间）
    """
    # 归一化 embedding
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # 计算余弦相似度
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

add_before_save("<id>382</id><title>中国下一代战机真的就这么来了外媒惊呼震撼</title><event>事件抽取的结果</event><action>的观点</action>Nan实体抽取的结果:Nan")