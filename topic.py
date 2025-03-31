import numpy as np
import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')
class NewsTopicAnalyzer:
    def __init__(self, news_data, num_topics=10):
        """
        初始化新闻主题分析器

        参数:
        - news_data: 新闻文本列表
        - num_topics: 主题数量
        """
        self.news_data = news_data
        self.num_topics = num_topics
        self.processed_docs = []

    def preprocess_text(self, text):
        """
        文本预处理

        1. 删除特殊字符
        2. 分词
        3. 去除停用词
        """
        # 删除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)

        # 分词
        words = list(jieba.cut(text))

        # 停用词列表
        stop_words = set([
            '的', '了', '和', '是', '都', '在', '以', '及', '与',
            '等', '很', '可以', '会', '有', '某', '这', '那', '之'
        ])

        # 过滤停用词和短词
        words = [word for word in words
                 if word not in stop_words and len(word) > 1]

        return ' '.join(words)

    def prepare_data(self):
        """
        准备文本数据
        """
        # 预处理所有文档
        self.processed_docs = [
            self.preprocess_text(doc) for doc in self.news_data
        ]

    def extract_topics(self):
        """
        使用LDA提取主题
        """
        # 构建文档-词矩阵
        vectorizer = CountVectorizer(max_df=0.65, min_df=2)
        doc_word_matrix = vectorizer.fit_transform(self.processed_docs)

        # 词汇表
        vocab = vectorizer.get_feature_names_out()

        # LDA主题模型
        lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,  # 主题数量
            random_state=42,
            learning_method='online',
            learning_offset=50.,
            max_iter=10
        )

        # 训练模型
        lda_output = lda_model.fit_transform(doc_word_matrix)

        # 存储结果
        self.lda_model = lda_model
        self.vocab = vocab
        self.lda_output = lda_output

        return lda_model, vocab, lda_output

    def return_top_words(self, n_top_words=4):
        """
        打印每个主题的top关键词
        """
        feature_names = self.vocab

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_idx]

            #print(f"主题 {topic_idx + 1}:")
            return " ".join(top_features)


    def visualize_topics(self):
        """
        可视化主题分布
        """
        plt.figure(figsize=(12, 6))

        # 主题分布直方图
        topic_distribution = self.lda_output.sum(axis=0)
        plt.bar(range(self.num_topics), topic_distribution)
        plt.title('新闻主题分布')
        plt.xlabel('主题编号')
        plt.ylabel('文档数量')
        plt.show()

    def get_dominant_topics(self):
        """
        获取每篇文档的主导主题
        """
        dominant_topics = np.argmax(self.lda_output, axis=1)

        # 统计每个主题的文档数
        topic_counts = pd.Series(dominant_topics).value_counts()

        return {
            'dominant_topics': dominant_topics,
            'topic_counts': topic_counts
        }
