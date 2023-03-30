from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("ywan/unite-mup")
# model = AutoModelForMaskedLM.from_pretrained("ywan/unite-mup")

# tokenizer.save_pretrained(save_directory='transformers/unite-mup')
# model.save_pretrained(save_directory='transformers/unite-mup')

# tokenizer = AutoTokenizer.from_pretrained("transformers/unite-mup")
# model = AutoModelForMaskedLM.from_pretrained("transformers/unite-mup")
import sys
import time
import yaml
# sys.path.append('COMET_mello')
sys.path.append('UniTE_mello')
# sys.path.append('bert_score')
# cp1 = sys.modules.copy()
# sys.path.remove('COMET_mello')
# sys.path = ['/usr/bin', '/opt/tiger/fake_arnold/fairseq_mrt', '/opt/tiger/arnold_toolbox', '/usr/lib/python37.zip', '/usr/lib/python3.7', '/usr/lib/python3.7/lib-dynload', '/home/tiger/.local/lib/python3.7/site-packages', '/opt/tiger/fake_arnold/fairseq_mrt/bleurt', '/usr/local/lib/python3.7/dist-packages', '/usr/lib/python3/dist-packages', 'bert_score', '/tmp/tmpue384s4i', 'UniTE_mello']
from unite_comet.models import load_from_checkpoint

# cp2 = sys.modules.copy()

# # print('==================')
# # print(cp2.keys() - cp1.keys())   # 太多了

# sys.path.remove('COMET_mello')

# #sys_modules_backup = dict()
# #sys_modules_backup['comet'] = sys.modules['comet']

# import_modules = ['comet']
# import_modules += [s+'.ttypes' for s in import_modules]

# for k in import_modules:
#     if k in sys.modules.keys():
#         del sys.modules[k]

# sys.path.insert(0, 'UniTE_mello')

from unite_comet.models.regression.regression_metric import RegressionMetric


model_prefix = 'UniTE-models/UniTE-MUP/'
model_path = model_prefix + 'checkpoints/UniTE-MUP.ckpt'
hparams_ref_path = model_prefix + 'hparams.ref.yaml'

info = 'ref'   # ref or src_ref

with open(hparams_ref_path) as yaml_file:
    hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    if 'src' in info: hparams['input_segments'] = ['hyp', 'src', 'ref']

model = RegressionMetric.load_from_checkpoint(model_path, **hparams)
model.eval()


srcs = ['a']
refs = ['Theoretisch ist dies allerdings möglich!']
hyps = ['The hotel staff was extremely friendly and helpful, rooms were clean and comfortable and location was extremely convenient.']

data = {'src': srcs, 'mt': hyps, 'ref': refs}
data = [dict(zip(data, t)) for t in zip(*data.values())]

predictions = model.predict_mello(samples=data, batch_size=1, device=1)
print(predictions)

# python3 mello_scripts/metrics_test/unite_test/unite_test.py
