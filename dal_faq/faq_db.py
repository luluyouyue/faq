# coding:utf-8
import os
import preprocess
from seg.seg import WordSeger
from config import config
from frame import logger

from .service import get_all_faqs_from_oncall_server

cur_dir = os.path.dirname(os.path.abspath(__file__))


def init_faq_corpus():
    faqs = get_all_faqs_from_oncall_server()
    faq_path = os.path.join(cur_dir, '../', 'faq', 'data/questions')
    faq_seged_path = os.path.join(cur_dir, '../', 'faq', 'data/seged_questions')
    faq_data_path = os.path.join(cur_dir, '../', 'faq', 'data/*')

    try:
        os.system('rm ' + faq_data_path)
    except AssertionError:
        pass

    with open(faq_path, 'w') as faq_writer, open(faq_seged_path, 'w') as faq_seged_writer:
        wordseger = WordSeger()
        faq_list = []
        sents = []
        for faq in faqs:
            key = str(faq.id).strip()
            query = faq.query.encode('utf-8').replace('\n', '。')
            candidate_query = faq.queryList.encode('utf-8').strip()

            query_list = [query]
            if candidate_query != '':
                query_list.extend(candidate_query.split('\n'))

            try:
                topic = faq.topic.lower()
                if topic == '':
                    topic = 'passport'
            except:
                topic = 'passport'

            faq_writer.write(key + '\x01' + '\x04'.join(query_list) + '\x01' + topic + '\n')
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


