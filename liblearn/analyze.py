# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:02:12 2013

@author: root
"""
import results
from os.path                import join
from liblearn.utils.analyze import analyze

objectives = ['convnet.error.final_test', 'convnet.error.final_valid']
substitutes = {}
ignores = ['model.layers']
ignores_val = None #{'features.learn_imbalance':0}
out_to_hp = ['convnet.nb_epoch']

experiment_path = join(results.path(), 'catsanddogs', 'exp001')
output_path = join(results.path(), 'catsanddogs', 'exp001_analysis')
analyze(experiment_path, output_path, objectives, ignores, ignores_val, substitutes, out_to_hp)
print('Done!')