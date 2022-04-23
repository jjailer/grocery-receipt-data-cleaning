import numpy as np
import pandas as pd
import datetime
import gc

import itertools
from tqdm import tqdm

import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-50")

