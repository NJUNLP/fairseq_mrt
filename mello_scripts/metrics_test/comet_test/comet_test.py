import sys
sys.path.append("COMET_mello")
# sys.path.append("bert_score")
from comet import load_from_checkpoint

# model_path = download_model("wmt20-comet-da")   # /home/tiger/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt
model_path = 'COMET_mello/checkpoints/wmt20-comet-da.ckpt'
model = load_from_checkpoint(model_path)

src = '"The Government has come up with an emergency rescue package we believe will address the unique challenges faced by the state," Ms Plibersek said today.'
hyp = '"Die Regierung hat ein Notfallrettungspaket vorgelegt, von dem wir glauben, dass es ihnen zugute auch Honig wird verneint.'
ref = '"Die Regierung hat ein Notfallrettungspaket geschnürt, das unserer Ansicht nach die einzigartigen Herausforderungen angeht, denen sich der Bundesstaat gegenüber sieht", erklärte Plibersek heute.'

data = {'src': [src], 'mt': [hyp], 'ref': [ref]}
# data = {'src': ['That, in fact, is what makes it different from advertising.'], 'mt': ['Das macht sie in der Tat anders als Werbung.'], 'ref': ['Dies macht den Unterschied der Information zur Werbung aus.']}
data = [dict(zip(data, t)) for t in zip(*data.values())]

seg_scores = model.predict_mello(data, batch_size=8, device=1)

print(seg_scores)
"""


"""

# python3 mello_scripts/metrics_test/comet_test/comet_test.py
