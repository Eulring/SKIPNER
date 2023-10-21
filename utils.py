import ipdb

def parse_span(res):
    res = res.replace("Output: ", "")
    nwords = res.split(" ")
    pl, pr = [], []
    for i, word in enumerate(nwords):
        if "@@" in word: pl.append(i)
        if "##" in word: pr.append(i)
    spans = []
    for l_, r_ in zip(pl ,pr):
        spans.append([l_, r_])
    return spans


class DL_metric(object):
    def __init__(self):
        self.pred_count_bel = 0
        self.gold_count_bel = 0
        self.match_count_bel = 0
        self.pred_count_ebd = 0
        self.gold_count_ebd = 0
        self.match_count_ebd = 0

    def update(self, step, idx, num=1):
        if step == 1:
            if idx == 'pred':
                self.pred_count_ebd += num
            if idx == 'gold':
                self.gold_count_ebd += num
            if idx == 'match':
                self.match_count_ebd += num
        if step == 2:
            if idx == 'pred':
                self.pred_count_bel += num
            if idx == 'gold':
                self.gold_count_bel += num
            if idx == 'match':
                self.match_count_bel += num

    def vis(self):
        print(self.pred_count_bel)
        print(self.gold_count_bel)
        print(self.match_count_bel)

    def compute(self):
        try: ebd_precision = self.match_count_ebd / self.pred_count_ebd
        except: ebd_precision = 0
        try: ebd_recall = self.match_count_ebd / self.gold_count_ebd
        except: ebd_recall = 0
        if ebd_precision + ebd_recall == 0: ebd_f1 = 0
        else: ebd_f1 = 2 * ebd_precision * ebd_recall / (ebd_precision + ebd_recall)

        try: bel_precision = self.match_count_bel / self.pred_count_bel
        except: bel_precision = 0
        try: bel_recall = self.match_count_bel / self.gold_count_bel
        except: bel_recall = 0
        if bel_precision + bel_recall == 0: bel_f1 = 0
        else: bel_f1 = 2 * bel_precision * bel_recall / (bel_precision + bel_recall)

        try: precision_all = self.match_count_bel / self.pred_count_bel
        except: precision_all = 0
        try: recall_all = self.match_count_bel / self.gold_count_ebd
        except: recall_all = 0
        if precision_all + recall_all == 0: f1_all = 0
        else: f1_all = 2 * precision_all * recall_all / (precision_all + recall_all)

        print("EBD pre = {}".format(ebd_precision))
        print("EBD rec = {}".format(ebd_recall))
        print("EBD f1 = {}".format(ebd_f1))
        print()
        print("BEL pre = {}".format(bel_precision))
        print("BEL rec = {}".format(bel_recall))
        print("BEL f1 = {}".format(bel_f1))
        print()
        print("Pipeline pre = {}".format(precision_all))
        print("Pipeline rec = {}".format(recall_all))
        print("Pipeline f1 = {}".format(f1_all))
        print()

        return precision_all, recall_all, f1_all
