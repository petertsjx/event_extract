import chromadb
from embedding import MyEmbeddingFunction,model
from utils import read_tag,compute_cosine_similarity
import numpy as np

client = chromadb.PersistentClient(path="./save/to")

class Query:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./save/to")
        self.collections={}
        self.tags=[]
        return

    def classify(self,text,tag):
        res=read_tag(tag,text)
        if (res == []):
            return res
        name = tag.strip('<>')
        self.collections[name] = client.get_or_create_collection(name=name, embedding_function=MyEmbeddingFunction())
        results = self.collections[name].query(
            query_texts=res,
            n_results=10
            )
        return results

    def add_to_chroma(self,data,ids,tag):
        name=tag.strip('<>')
        self.collections[name]=client.get_or_create_collection(name=name, embedding_function=MyEmbeddingFunction())
        self.collections[name].add(
                documents=data,
                ids = ids
            )
        print(name,"count:",self.collections[name].count())

    def set_tag(self,tags):
        self.tags=tags

    def get_tags(self):
        return self.tags

    def test_count(self):
        for tag in self.tags:
            name = tag.strip('<>')
            self.collections[name] = client.get_or_create_collection(name=name,
                                                                     embedding_function=MyEmbeddingFunction())
            print(name,self.collections[name].count())

    def predict_distance(self,result,filter):
        keys = np.array(result["ids"]).flatten()
        values = np.array(result["distances"]).flatten()
        result_dict = dict(zip(keys, values))
        # 获取值小于200的元素
        filtered_dict = {k: v for k, v in result_dict.items() if v < filter}
        # 返回满足条件的值
        return filtered_dict

    def search_documents_by_id(self,ids,name):
        query_collection=client.get_or_create_collection(name=name, embedding_function=MyEmbeddingFunction())
        results = query_collection.get(ids=ids)
        return results['documents'],results["ids"]

    def caculate_entity_similariy(self,text1,text2):
        return compute_cosine_similarity(model.encode(text1), model.encode(text2))

    def delete_collection(self,name):
        try:
            self.client.delete_collection(name=name)
            print(f"Collection '{name}' 删除成功")
        except Exception as e:
            print(f"删除 Collection 失败: {e}")


