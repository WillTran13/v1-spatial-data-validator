# threshold came from the clean coco8 distribution
# by setting below the observed clean min with -30% margin


THRESHOLDS = {
    "blur_min": 600,  # clean min 834.7 (range: 834-27971, big spread)
    "brightness_min": 70,  # clean min 101.6
    "brightness_max": 220,  # clean max 143.3, set the ceiling below the clipping zone
    "contrast_min": 35,  # clean min 50.7
    "resolution_min": 300,  # clean min 381
}

# see the result below from sweep_experiment.py
# the crossing lands at severity of 0.5, for both metric
# measured in 4 coco8 frames (training).

SIGMA_MAX = 1.3
# Each frame crossed blur_min=600 at signma .45/.55/.75/.8
# So middle value is .65, double the pass fail transition at severity of 0.5 instead at the end of the ramp.

ALPHA_MIN = 0.25
# each frame cross min brightness=70 at alpha .53/.59/.65/.69
# middle .62, dies form 1 doubled.