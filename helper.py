import geotechnipy.chamber as chamber
import geotechnipy.cptc as cptc
import geotechnipy.soil as soil
import numpy as np
import pandas as pd
import pickle

def get_cresistance(
    cid
):
    """
    Get the file containing cone resistance data.

    Parameters
    ----------
    cid:  str | calibration chamber CPT ID

    Returns
    -------
    cresistance:  tuple | tuple in the form ((readings),
        (reading depths))

    """
    # Unpickle the sieve analysis data.
    with open(
        'cresistance/{}.p'.format(cid),
        'rb'
    ) as in_file:
        cresistance = pickle.load(in_file)

    return cresistance


def get_sresistance(
    cid
):
    """
    Get the file containing sleeve resistance data.

    Parameters
    ----------
    cid:  str | calibration chamber CPT ID

    Returns
    -------
    cresistance:  tuple | tuple in the form ((readings),
        (readings depths))

    """
    # Unpickle the sieve analysis data.
    with open(
        'sresistance/{}.p'.format(cid),
        'rb'
    ) as in_file:
        sresistance = pickle.load(in_file)

    return sresistance
