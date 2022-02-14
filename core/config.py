# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

cfg.GPU_ID = '0'
cfg.EXP = 'anet12'
cfg.LR = '[0.0001]*100000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_CLASSES = 100
cfg.MODAL = 'all'
cfg.FEATS_DIM = 2048
cfg.BATCH_SIZE = 256
cfg.DATA_PATH = './data/ActivityNet12'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_EASY = 10
cfg.R_HARD = 8
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 500
cfg.PRINT_FREQ = 100
cfg.CLASS_THRESH = 0.1
cfg.NMS_THRESH = 0.6
cfg.CAS_THRESH = np.arange(0.0, 0.15, 0.015)
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
cfg.TIOU_THRESH = np.linspace(0.5, 0.95, 10)
cfg.UP_SCALE = 24
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'gt.json')
cfg.SEED = 0
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 50
cfg.CLASS_DICT = {'Archery': 0, 'Ballet': 1, 'Bathing dog': 2, 'Belly dance': 3, 'Breakdancing': 4, 'Brushing hair': 5, 'Brushing teeth': 6, 'Bungee jumping': 7, 'Cheerleading': 8, 'Chopping wood': 9, 'Clean and jerk': 10, 'Cleaning shoes': 11, 'Cleaning windows': 12, 'Cricket': 13, 'Cumbia': 14, 'Discus throw': 15, 'Dodgeball': 16, 'Doing karate': 17, 'Doing kickboxing': 18, 'Doing motocross': 19, 'Doing nails': 20, 'Doing step aerobics': 21, 'Drinking beer': 22, 'Drinking coffee': 23, 'Fixing bicycle': 24, 'Getting a haircut': 25, 'Getting a piercing': 26, 'Getting a tattoo': 27, 'Grooming horse': 28, 'Hammer throw': 29, 'Hand washing clothes': 30, 'High jump': 31, 'Hopscotch': 32, 'Horseback riding': 33, 'Ironing clothes': 34, 'Javelin throw': 35, 'Kayaking': 36, 'Layup drill in basketball': 37, 'Long jump': 38, 'Making a sandwich': 39, 'Mixing drinks': 40, 'Mowing the lawn': 41, 'Paintball': 42, 'Painting': 43, 'Ping-pong': 44, 'Plataform diving': 45, 'Playing accordion': 46, 'Playing badminton': 47, 'Playing bagpipes': 48, 'Playing field hockey': 49, 'Playing flauta': 50, 'Playing guitarra': 51, 'Playing harmonica': 52, 'Playing kickball': 53, 'Playing lacrosse': 54, 'Playing piano': 55, 'Playing polo': 56, 'Playing racquetball': 57, 'Playing saxophone': 58, 'Playing squash': 59, 'Playing violin': 60, 'Playing volleyball': 61, 'Playing water polo': 62, 'Pole vault': 63, 'Polishing forniture': 64, 'Polishing shoes': 65, 'Preparing pasta': 66, 'Preparing salad': 67, 'Putting on makeup': 68, 'Removing curlers': 69, 'Rock climbing': 70, 'Sailing': 71, 'Shaving': 72, 'Shaving legs': 73, 'Shot put': 74, 'Shoveling snow': 75, 'Skateboarding': 76, 'Smoking a cigarette': 77, 'Smoking hookah': 78, 'Snatch': 79, 'Spinning': 80, 'Springboard diving': 81, 'Starting a campfire': 82, 'Tai chi': 83, 'Tango': 84, 'Tennis serve with ball bouncing': 85, 'Triple jump': 86, 'Tumbling': 87, 'Using parallel bars': 88, 'Using the balance beam': 89, 'Using the pommel horse': 90, 'Using uneven bars': 91, 'Vacuuming floor': 92, 'Walking the dog': 93, 'Washing dishes': 94, 'Washing face': 95, 'Washing hands': 96, 'Windsurfing': 97, 'Wrapping presents': 98, 'Zumba': 99}
