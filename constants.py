NEY_WAV_DIR = "dataset/ney/"
GTR_WAV_DIR = "dataset/gtr/"
NEY_FEATURE_DIR = "dataset/features/ney/"
GTR_FEATURE_DIR = "dataset/features/gtr/"

SR = 48000
N_FFT = 512
HOP = 64
WINDOW_SAMPLE_LEN = 2 ** 14 - HOP
SIGNAL_HOP = 2 ** 10

# for test time
OVERLAP = 2 ** 8
