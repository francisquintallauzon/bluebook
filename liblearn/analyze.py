# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:02:12 2013

@author: root
"""

from learn.utils.analyze    import analyze

objectives = ['emneuron.error.valid_smooth']
substitutes = {"preprocess.nb_pca_components":"preprocess.nb_pca", "preprocess.np_pca":"preprocess.nb_pca"}
ignores = ['dataset.data_path']
ignores_val = None #{'features.learn_imbalance':0}
out_to_hp = ['zerobias_0.nb_steps', 'emneuron.nb_steps']

experiment_path = '/home/francis/experiments/results/emneuron_014'
output_path = '/home/francis/experiments/results/emneuron_014_analysis'
analyze(experiment_path, output_path, objectives, ignores, ignores_val, substitutes, out_to_hp)
print 'Done!'