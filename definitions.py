import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPIREMENTS_DIR = os.path.join(ROOT_DIR, 'expirements')

ISIS_2016_DIR = os.path.join(DATA_DIR, 'isic2016')
ISIS_2019_DIR = os.path.join(DATA_DIR, 'isic2019')
ISIS_2024_DIR = os.path.join(DATA_DIR, 'isic2024')

SEGMENTED_ISIC_2024_DIR = os.path.join(DATA_DIR, 'segmented_isic_2024')

EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'ISIC-images-mal_cropped')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')