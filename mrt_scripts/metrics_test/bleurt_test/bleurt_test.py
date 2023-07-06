import sys
sys.path.append('bleurt')
# hyps = ['Room seemed clean and confortable .The room was clean and comfortable .- The room appeared clean and comfortable .- The room appeared clean and comfortable. The room was clean and confortable.']
hyps = ["Lage vom Hotel war grundsätzlich bestens − Hotelpersonal weitgehend zuvorkommend bzw. ggf. hilfehilfsbereit. Vor allem die Lage des Hotels war gut, Hotelmitarbeiter grundsätzlich äußerst lieb bzw. gegebenenfalls auch durchaus hilfehilfsbereit."]
refs = ['May the sunshine always be with you.']

# ============================== 第一种调用 ============================== #
# from bleurt import score
# bleurt_checkpoint = "bleurt/bleurt-large-512"
# bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)
# scores = bleurt_scorer.score(references=refs, candidates=hyps)
# print('hypo: ' + hyps[0])
# print('ref: ' + refs[0] + '      ' + 'bleurt score: ' + str(scores[0]))

from bleurt import score

checkpoint = "bleurt/BLEURT-20"
# refs = ["This is a test."]
# hyps = ["This is the test."]

# scorer = score.BleurtScorer(checkpoint)
# print('=============================')
# print(scorer)
# scores = scorer.score(references=refs, candidates=hyps)
# assert isinstance(scores, list) and len(scores) == 1
# print(scores)

# ============================== 第二种调用 ============================== #

import xmlrpc.client
device_id = int(sys.argv[1])
proxy = xmlrpc.client.ServerProxy('http://localhost:8888')
scores = proxy.bleurt_score((hyps, refs), device_id)
print("===================")
print(scores)

# python3 mrt_scripts/metrics_test/bleurt_test/bleurt_test.py 0
