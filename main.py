from utils import *
from chat import ChatBot
from query import Query
from topic import NewsTopicAnalyzer



def prepare_query_data(path):
    df=process_full_test_data(path)
    content={}
    for index,row in df.iterrows():
        content[row['id']]=row['title']+"_!_"+row['description']
    return content

def prompt_start():
    return '''
    1）抽取出以下新闻中的事件,10-25字，如果无法给出答案,请填写Nan,填写到<event>标签内。
    2）并且根据抽取出的事件自行总结出一个事件动作分类（十个字以内），如果无法给出答案,请填写Nan,填写到<action>标签内，
    3）抽取出新闻中的实体和重点词（人名、地名、机构名、专业名词、国家名等等），如果无法给出答案,请填写Nan,填写到<entity>标签内。

    需要以这样的格式给出答案: 
    <event>事件抽取的结果</event>
    <action>事件动作分类的结果 </action>
    <entity>实体抽取的结果</entity>     
    \n
    '''

def cluster_prompt_start():
    return '''
    请给下列文字总结出一个主题，10-20字之间,要突出实体，答案写在：
    <topic></topic>之间。
    '''

def insert_data_into_db(data_path,tags,query):
    for tag in tags:
        data,ids=read_data_for_chroma(data_path,tag)
        query.add_to_chroma(data,ids,tag)



def main():
    prediction=True
    construct_db_data=False
    insert_into_db=False
    test_count=False
    delete_collection=False
    cluster_topic=True

    entity_ratio=0.8
    tags = ['<title>', '<event>', '<action>', '<entity>']
    topic={}
    topic_ids={}
    topic_count=0
    query = Query()
    answers = []
    if prediction and not cluster_topic:
        if not cluster_topic:
            data_path ="../data/final_full_test.csv"
        else:
            data_path="./final_examples.txt"
        bot = "deepseek-v2:16b"
        #prepare data
        result=prepare_query_data(data_path)

        ##chat with Bot
        prompt = prompt_start()
        chat_bot = ChatBot(prompt, bot)

        for id,content in result.items():
            res=chat_bot.run([content])
            chat_answer="<id>"+str(id)+"</id>"+"<title>"+result[id].split("_!_")[0]+"</title>"+res[0]
            chat_answer = add_before_save(chat_answer)
            answers.append(chat_answer)
    elif cluster_topic:
        with open('final_examples.txt', 'r', encoding='utf-8') as f:
            for line in f:
                answers.append(line)

    ##query with chromadb
        for answer in answers:
            distance_dict={}
            id_count=0
            final_ids = []
            final_documents = []
            id_list=[]
            for tag in tags:
                result=query.classify(answer,tag)
                #返回distance
                if result!=[]:
                    distance_dict[tag]=query.predict_distance(result,300)
            #distance<400所有id进行下一步分析
            for _,value in distance_dict.items():
                for id,_ in value.items():
                    id_list.append(id)
            if id_list==[]:
                print("no category, maybe a new class ")
            else:
                #去重，重复的id说明可能性高。
                deduplicate_result = deduplicate_with_duplicates(id_list)
                #计算实体相似度

                documents,ids=query.search_documents_by_id(deduplicate_result['unique_list'],"entity")
                for d,id in zip(documents,ids):
                    if read_tag("<entity>",answer)!=[]:
                       sim=query.caculate_entity_similariy(read_tag("<entity>",answer)[0],d)
                       if sim>entity_ratio:
                            final_ids.append(id)
                            final_documents.append(d)
                #显示分类结果
                if len(final_ids)>0:
                    documents, ids = query.search_documents_by_id(final_ids, "title")
                    #print(read_tag('<title>',answer))
                    #print(documents)
                if cluster_topic:
                    if(len(final_ids)>3):
                        print("可能出现主题分类")
                        '''
                        #LLM方法 todo,
                        text=read_tag('<title>', answer)[0]+"\n"
                        for i,d in enumerate(documents):
                            text +=d+"\n"
                        
                        cluster_bot = ChatBot(cluster_prompt_start(), bot)
                        cluster_answer = cluster_bot.run([text])
                        print(cluster_answer[0])
                        '''
                        #LDA
                        documents.append(read_tag('<title>', answer)[0])
                        analyzer=NewsTopicAnalyzer(documents, num_topics=2)
                        analyzer.prepare_data()

                        # 提取主题
                        #lda_model, vocab, lda_output = analyzer.extract_topics()
                        #topic_results = analyzer.get_dominant_topics()
                        #print(topic_results)
                        #topic_name=analyzer.return_top_words()
                        topic_count+=1
                        #topic[topic_count]=topic_name
                        topic_ids[topic_count]=final_ids
                        print(answer)
                        print(documents,final_ids)
                        print()
        print(topic_count,topic,topic_ids)




    if construct_db_data:
        data_path = "../data/full_test.csv"
        bot = "deepseek-v2:16b"
        # prepare data
        result = prepare_query_data(data_path)

        ##chat with Bot
        prompt = prompt_start()
        chat_bot = ChatBot(prompt, bot)
        with open('final_examples.txt', 'w', encoding='utf-8') as f:
            for id, content in result.items():
                if len(content.split("_!_")[0])<2:
                    print(id,"bad data, ignoring")
                    chat_answer = "<id>" + str(id) + "</id>"+"<bad>"
                    f.write(chat_answer)
                else:
                    res = chat_bot.run([content])
                    chat_answer = "<id>" + str(id) + "</id>" + "<title>" + result[id].split("_!_")[0] + "</title>" + res[0]+"\n"
                    chat_answer = add_before_save(chat_answer)
                    print(chat_answer)
                    f.write(chat_answer)

    if insert_into_db:
        data_path="final_examples.txt"
        query.set_tag(tags)
        insert_data_into_db(data_path, tags,query)

    if test_count:
        query.set_tag(tags)
        query.test_count()

    if delete_collection:
        for tag in tags:
            name = tag.strip('<>')
            query.delete_collection(name)


if __name__ == "__main__":
    main()