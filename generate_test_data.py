# -*- coding: utf-8 -*-


def generate_pairs(querys, ll):
    if len(querys) <= 1:
        return
    else:
        querys = list(querys)
        first = querys[0]
        for second in querys[1:]:
            ll.append((first, second))
        generate_pairs(querys[1:], ll)


if __name__ == "__main__":
    fin = open('faq/data/test', 'r')
    fout = open('faq/data/test_zuhe', 'w')
    qq_similarity = {}
    for line in fin.readlines():
        items = line.strip().split('\t')
        id = items[0]
        if id not in qq_similarity:
            querys = set()
        else:
            querys = qq_similarity[id]

        for query in items[1:]:
            if query not in ['passport', 'push'] and query != '':
                querys.add(query)
        # if id == '1002':
        #     print 'id querys = ', ' '.join(querys)
        qq_similarity[id] = querys

    for id, querys in qq_similarity.items():
        print 'id = ', id, 'querys = ', ' '.join(querys)
        if len(querys) <= 1:
            continue
        else:
            ll = []
            generate_pairs(querys, ll)

        for first, second in ll:
            fout.write(id + '\t' + first + '\t' + second + '\n')

