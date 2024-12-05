NEY_WAV_DIRS = ["dataset/ney_1/", "dataset/ney_2/"]
GTR_WAV_DIRS = ["dataset/gtr_1/", "dataset/gtr_2/"]
NEY_FEATURE_DIR = "dataset/features/ney/"
GTR_FEATURE_DIR = "dataset/features/gtr/"

SR = 48000
N_FFT = 511 * 2
HOP = 64
WINDOW_SAMPLE_LEN = 2 ** 15 - HOP
SIGNAL_HOP = 2 ** 10

# for test time
OVERLAP = 2 ** 8
