import numpy as np
from rlpy.Tools import deltaT, clock, hhmmss, getTimeStr
import os

"""
GridWorld Random reward generator
"""

def generate(mappath, goals=2, save=False, savepath="reward_locations"):
    """
    Returns a numpy array of goal positions (without repeats nor ordering).
    Goal positions are picked from empty spots on map.

    :param mappath: Path to GridWorld Map
    :param goals: Number of goals placed
    :param save: Boolean for saving array for repeatability
    """

    map_arr = np.loadtxt(mappath, dtype=np.uint8)
    possibles = np.argwhere(map_arr == 0)
    sample_indices = np.random.choice(len(possibles), goals, replace=False)
    goals = possibles[sample_indices]
    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        path = os.path.join(savepath, getTimeStr())
        np.savetxt(path, goals, fmt="%1d")
    return goals


