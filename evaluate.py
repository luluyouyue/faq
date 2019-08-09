# -*- coding: utf-8 -*-
import faq
import preprocess
import os
from config import config
from seg.seg import WordSeger
cur_path = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    ## 设置测试的faq
    test_questions = os.path.join(cur_path, "faq", "test_data", "test_questions")
    with open(test_questions, 'r') as fin:
            wordseger = WordSeger()
            faq_list = []
            sents = []
            for line in fin.readlines():
                line = line.strip()
                try:
                    id, raw_answers, answer_type = line.split('\x01')
                    key = int(id.strip())
                except:
                    continue

                query_list = raw_answers.strip().split('\x04')

                try:
                    topic = answer_type.lower()
                    if topic == '':
                        topic = 'passport'
                except:
                    topic = 'passport'

                for ori_query in query_list:
                    new_query = preprocess.rewrite(ori_query)
                    if new_query in sents:
                        continue
                    else:
                        sents.append(new_query)
                    clean_words = wordseger.get_wordseg(new_query)
                    t = (int(key), clean_words, ori_query, topic)
                    faq_list.append(t)

            config.set_value("faq_list", faq_list)

    with open(os.path.join(cur_path, 'faq/test_data/test.faq.2'), 'r') as fin, open(os.path.join(cur_path, 'faq/test_data/test.faq.2_out'), 'w') as fout:
        faqer = faq.Faq()
        total = 0
        top3_right = 0
        top1_right = 0
        for line in fin.readlines():
            total += 1
            print 'total = ', total

            try:
                id, query = line.strip().split('\t')
            except:
                continue

            ids = id.strip().split('|')
            sent1 = preprocess.rewrite(query)
            answers = faqer.top_n_similariy_questions(sent1, 3, "test")

            top3_id = []
            for answer in answers:
                top3_id.append(str(answer.AnswerId))
            top3_is_true = 'top3_false'
            top1_is_true = 'top1_false'
            if len(set(ids) & set(top3_id)) != 0:
                top3_right += 1
                top3_is_true = 'top3_true'
            # print 'set(ids) = ', set(ids)
            # print 'list(top3_id[0] = ', list(top3_id[0])
            print 'top3_id = ', top3_id
            print 'top3_id[0] = ', set(list(top3_id[0]))
            if len(set(ids) & set([top3_id[0]])) != 0:
                top1_right += 1
                top1_is_true = 'top1_true'
            fout.write(id + '\t' + query + '\t' + '#'.join(top3_id) + '\t' + top3_is_true + '\t' + top1_is_true + '\n')
        print 'top3 precise rate = ', float(top3_right) / total
        print 'top1 precise rate = ', float(top1_right) / total
        fout.write('top3 precise rate = ' + '\t' + str(float(top3_right) / total) + '\n')
        fout.write('top1 precise rate = ' + '\t' + str(float(top1_right) / total) + '\n')
