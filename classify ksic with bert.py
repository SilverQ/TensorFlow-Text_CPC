import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

json_data = '/home/hdh/PycharmProjects/KoBERT-master/data/ksic00.json'
# train_data =
# test_data = '/content/drive/MyDrive/Deep Learning/datasets/ratings_test.txt'
# df_train = pd.read_csv(train_data, sep='\t')
# df_test = pd.read_csv(test_data, sep='\t')
df_raw = pd.read_csv(json_data)

df_raw.head()
