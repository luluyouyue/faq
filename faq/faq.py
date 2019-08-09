# -*- coding: utf-8 -*-
import os
import preprocess
from qq_similarity.qq_similarity import compare
# from qq_similarity.qq_similarity import compare_by_sentence_vector
# from qq_similarity.qq_similarity import BM25
from qq_similarity.qq_similarity import nearby
from qq_similarity.rule_based_qq_similarity import find_answer_by_rule
import questionRecall
from seg.seg import WordSeger
from passport_lark_bot_nlp_pyrpc.thrift_gen.toutiao.passport.lark_bot_nlp.ttypes import *
from frame import logger
import time
from config import config
import dal_faq
import threading
from threading import _Timer

# 定义的一些常量
cur_dir = os.path.dirname(os.path.abspath(__file__))

# 同义词典路径
DEFAULT__SYNONYM = os.path.join(cur_dir, "data", "word_synonym")

# 从数据库中初始化questions
dal_faq.init_faq_corpus()
refresh_interval = 600
is_test = False


class RepeatingTimer(_Timer):
    stop = False

    def run(self):
        while not self.finished.is_set():
            if RepeatingTimer.stop:
                self.finished.set()
            self.finished.wait(self.interval)
            self.function(*self.args, **self.kwargs)


class Faq(object):
    error = 0

    def __init__(self):
        self.faqs = config.get_value("faq_list")  # key: id, value: doc:
        self.synonym = dict()
        self.defualt_match_type = None
        self.load_defualt_match_type()
        start = time.time()
        self.init_synonym()
        end = time.time()
        logger.info("load faq segment used %d second, total Len faq = %d", end - start, len(self.faqs))
        logger.info("load synonym successful, synonym len = %d", len(self.synonym))

        if not is_test:
            self.repeatingTimer = RepeatingTimer(refresh_interval, self.update_dict)
            self.repeatingTimer.start()

    def init_synonym(self, num=10):
        '''
        :param num: number of sysnonyms
        '''
        with open(DEFAULT__SYNONYM, 'w') as fout:
            for _, words, _, _ in self.faqs:
                for word in words:
                    if word not in self.synonym:
                        nearby_words = nearby(word)[0]
                        if len(nearby_words) > 10:
                            nearby_words = nearby_words[:10]
                        self.synonym[word] = nearby_words
                        fout.write(word + '\x01' + '\x04'.join(nearby_words) + '\n')

    def load_defualt_match_type(self):
        conf = config.get_value("conf")
        if conf is None:
            logger.error("load lark_bot_nlp.conf error!")
        try:
            all_support_type = conf["default_match_type"]["all_support_type"]
            all_support_type = map(lambda x: x.strip().lower(), all_support_type.strip('|').split('|'))
            defualt_match_type = conf["default_match_type"]["match_type"]
            defualt_match_type = map(lambda x: x.strip().lower(), defualt_match_type.strip('|').split('|'))

            # 判断默认匹配类型是否配错
            assert len(set(defualt_match_type) & set(all_support_type)) == len(set(defualt_match_type)), \
                'set defualt match type error'
        except:
            raise Exception("load defualt_match_type error, please set defualt_match_type in lark_bot_nlp.conf")
        self.defualt_match_type = defualt_match_type

    def top_n_similariy_questions(self, query, topN, req_faq_type):
        '''
        :param req_faq_type:
        :param query:
        :param topN:
        :return: list
        '''
        defualt_match_type_on = False
        # logger.info("origin query = %s", query)
        rewrite_query = preprocess.rewrite(query)
        if req_faq_type is None or req_faq_type == '':
            defualt_match_type_on = True
        try:
            req_faq_type = req_faq_type.lower()
        except:
            req_faq_type = 'passport'

        logger.info("len faqs = %d, req_faq_type = %s", len(self.faqs), req_faq_type)
        query_words = WordSeger.get_wordseg(rewrite_query)
        logger.info("query seged words = %s", '#'.join(query_words))
        score_dict = {}
        logger.info("query words = %s", ' '.join(query_words))
        filted_faqs = questionRecall.recall(self.faqs, query_words)
        logger.info("filted faqs length = %d", len(filted_faqs))
        rule_answers = find_answer_by_rule(query_words, filted_faqs)
        # print 'answer = ', rule_answers
        if len(rule_answers) != 0:
            for rule_answer in rule_answers:
                if len(rule_answer) != 2:
                    continue
                rule_key = rule_answer[0]
                score_dict[rule_key] = rule_answer[1]

        # bm25相似度计算例子
        # bm25er = BM25(filted_faqs)
        for key, words, ori_query, faq_type in filted_faqs:
            if req_faq_type == "test":
                pass
            else:
                if defualt_match_type_on:
                    if faq_type not in self.defualt_match_type:
                        continue
                elif req_faq_type not in faq_type:
                    continue
            qq_score = compare(query_words, words, self.synonym)
            # qq_score = compare_by_sentence_vector(query_words, words)
            # qq_score2 = bm25er.sim(query_words, words)
            # qq_score = qq_score1
            logger.info("score = %f, compare %s && %s", qq_score, '#'.join(query_words), '#'.join(words))
            if key in score_dict:
                origin_score = score_dict[key][0]
                if qq_score > origin_score:
                    score_dict[key] = (qq_score, ori_query, '-1', None)
            else:
                score_dict[key] = (qq_score, ori_query, '-1', None)
        logger.info("compare finished")

        # 所有的计算结果排序
        res = sorted(score_dict.items(), key=lambda x: x[1][0], reverse=True)[:topN]
        answers = []
        logger.info("res length = %d", len(res))
        for key, scored_answer in res:
            score, answer, match_type, rule = scored_answer
            logger.info("top n result: key = %s, score = %f, query = %s", key, score, answer)
            answer = Answer(AnswerId=key, Score=score, SimilarityQuery=answer, RewriteQuery=rewrite_query,
                            CleanWords='#'.join(query_words), MatchType=match_type, Rule=rule)
            answers.append(answer)
        return answers

    @property
    def error_code(self):
        return Faq.error

    def update_dict(self):
        '''
        定时更新faq、分词、同义词字典
        '''
        print '>>update dict, active thread count = ', threading.active_count()
        # 定时器,参数为(多少时间后执行，单位为秒，执行的方法)
        dal_faq.init_faq_corpus()
        self.init_synonym()
        self.faqs = config.get_value("faq_list")
