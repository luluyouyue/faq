# coding:utf-8
import os
import preprocess
from seg.seg import WordSeger
from config import config


cur_dir = os.path.dirname(os.path.abspath(__file__))

## 这块可以替换成从数据库读取faq
# def get_all_faqs_from_oncall_server(faq_path):
#     with open(faq_path, 'r') as fin:
#         lines = fin.readlines()


def init_faq_corpus():
    # faqs = get_all_faqs_from_oncall_server()
    faq_path = os.path.join(cur_dir, '../', 'faq', 'data/questions')
    faq_seged_path = os.path.join(cur_dir, '../', 'faq', 'data/seged_questions')
    faq_data_path = os.path.join(cur_dir, '../', 'faq', 'data/*')

    # try:
    #     os.system('rm ' + faq_data_path)
    # except AssertionError:
    #     pass

    faqs = [['1', '如何健康有效地减肥？', '如何减肥？', '怎么减肥？', '有什么快速的减肥攻略?', 'jianfei',
             \ '肥胖是指身体脂肪的比例过高。减肥的核心就是保持摄入能量小于消耗的能量，人体就会消耗脂肪产生能量。三分锻炼七分吃。首先要改善饮食，食用高营养低热量的食物。其次可以适当增加运动量。并且保持轻松心情和早睡早起。长期保持即可有效地减肥。']]
    with open(faq_path, 'w') as faq_writer, open(faq_seged_path, 'w') as faq_seged_writer:
        wordseger = WordSeger()
        faq_list = []
        sents = []
        for faq in faqs:
            key = str(faq[0]).strip()
            query_list = faq[1:6]
            for ori_query in query_list:
                new_query = preprocess.rewrite(ori_query)
                if new_query in sents:
                    continue
                else:
                    sents.append(new_query)
                clean_words = wordseger.get_wordseg(new_query)
                t = (int(key), clean_words, ori_query, topic)
                faq_list.append(t)
                faq_seged_writer.write(key + '\x01' + new_query + '\x01' + '\x04'.join(clean_words) + '\x01' + topic + '\n')
        config.set_value("faq_list", faq_list)

        del sents
        logger.info("init faq db Len faq = %d", len(faq_list))


