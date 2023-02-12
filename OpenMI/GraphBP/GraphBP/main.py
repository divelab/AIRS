from config import conf
from runner import Runner
import os


binding_site_range = 15.0


out_path = 'trained_model'
if not os.path.isdir(out_path):
    os.mkdir(out_path)

runner = Runner(conf, out_path=out_path)
runner.train(binding_site_range)