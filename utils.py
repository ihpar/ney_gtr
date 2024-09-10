import numpy as np
from constants import *


def stitch_wave_chunks(waves):
    """ 
    Takes wave chunks as a list and stitches them with cross-fades.
    """
    stitched_wave = np.array([])
    fade_in = np.linspace(0, 1, FADE_SAMPLE_LEN)
    fade_out = np.linspace(1, 0, FADE_SAMPLE_LEN)

    for i, wave in enumerate(waves):
        if i > 0:
            # fade in wave chunks after the 1st one
            wave[:FADE_SAMPLE_LEN] *= fade_in
        # fade out all wave chunks
        wave[-FADE_SAMPLE_LEN:] *= fade_out

        if len(stitched_wave) > 0:
            stitched_wave[-FADE_SAMPLE_LEN:] += wave[:FADE_SAMPLE_LEN]
            stitched_wave = np.append(stitched_wave, wave[FADE_SAMPLE_LEN:])
        else:
            stitched_wave = np.append(stitched_wave, wave)

    return stitched_wave
