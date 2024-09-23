import numpy as np
from constants import *


def stitch_wave_chunks(waves):
    """ 
    Takes wave chunks as a list and stitches them with cross-fades.
    """
    stitched_wave = np.array([])
    fade_in = np.linspace(0, 1, WINDOW_SLIDE_MARGIN)
    fade_out = np.linspace(1, 0, WINDOW_SLIDE_MARGIN)

    for i, wave in enumerate(waves):
        if i > 0:
            # fade in wave chunks after the 1st one
            wave[:WINDOW_SLIDE_MARGIN] *= fade_in
        # fade out all wave chunks
        wave[-WINDOW_SLIDE_MARGIN:] *= fade_out

        if len(stitched_wave) > 0:
            stitched_wave[-WINDOW_SLIDE_MARGIN:] += wave[:WINDOW_SLIDE_MARGIN]
            stitched_wave = np.append(
                stitched_wave, wave[WINDOW_SLIDE_MARGIN:])
        else:
            stitched_wave = np.append(stitched_wave, wave)

    return stitched_wave
