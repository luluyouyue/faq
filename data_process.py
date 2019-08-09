# -*- coding: utf-8 -*-
import os
import dal_faq
import faq

cur_path = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    # with open(os.path.join(cur_path, 'faq/test_data/test.faq'), 'r') as fin, open(os.path.join(cur_path, 'faq/test_data/test.faq.2'), 'w') as fout:
    #     all_test = []
    #     for line in fin.readlines():
    #         items = line.strip().split('\t')
    #         print 'len items = ', len(items)
    #         if len(items) < 3:
    #             continue
    #         id = items[0]
    #         for query in items[2:]:
    #             test_line = [id, query]
    #             all_test.append(test_line)
    #
    #     for item in all_test:
    #         fout.write('\t'.join(item) + '\n')
    dal_faq.get_all_questions()
