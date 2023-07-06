# import bert_score
# from bert_score import score

# hyps = ['great']
# refs = ['good']

# P, R, F1 = score(hyps, refs, lang='en', rescale_with_baseline=True)

import sys
sys.path.append("bert_score")
from bert_score import BERTScorer
scorer = BERTScorer(lang="de", rescale_with_baseline=True)

hyps = ['Das Personal ist sehr nett. Das Zimmer war groß.']
refs = ['good']

P, R, F1 = scorer.score(hyps, refs)
# print(P)
# print(R)
print(F1.tolist())


# python3 mrt_scripts/metrics_test/bertscore_test/bertscore_test.py