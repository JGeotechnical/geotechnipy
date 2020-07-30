import pickle
import csv
import geotechnipy.chamber as chamber
import geotechnipy.soil as soil
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sqlalchemy import (
    and_,
    between,
    Boolean,
    Column,
    create_engine,
    Float,
    insert,
    Integer,
    MetaData,
    select,
    String,
    Table,
    update,
)
import warnings


class CPTc():
    """Instantiate the Calibation Chamber Cone Penetration Test class."""

    def __init__(
        self,
        cid=None,
        sid=None,
        cdiameter=None,
        cresistance=None,
        sresistance=None,
        chpressure=None,
        cppressure=None,
        bcondition=None,
        hthstress=None,
        htvstress=None,
        htmstress=None,
        saturation=None,
        hppressure=None,
        hevstress=None,
        hehstress=None,
        hemstress=None,
        oratio=None,
        rdensity=None,
        rvcresistance=None,
        rvcresistance_mesnn=None,
        rvcresistance_n=None,
        rvcresistance_osnn=None,
        rvdeltaq=None,
        rvfratio=None,
        rvsresistance=None,
        sparameter=None,
        vratio=None
    ):
        """
        Instatiate the Calibration Chamber Cone Penetration Test base class.

        The user should note that if the parameter is a list it should be
        specified in the form [min, max].

        Parameters
        ----------
        cid:  str | Calibration chamber CPT ID
        cdiameter:  list | cone diameter (cm)
        cresistance:  bool | True if cone resistance (MPa) versus depth
            attached; False otherwise
        sresistance:  bool | True if sleeve resistance (kPa) versus depth
            attached; False otherwise
        cppressure:  bool | True if CPT pore pressure (kPa) versus depth
            attached; False otherwise
        chpressure:  bool | True if CPT lateral stress (kPa) versus depth
            attached; False otherwise
        rvcresistance:  float | representative value of cone resistance (MPa)
            for the test
        rvcresistance_mesnn:  float | representative value of mean effective
            stress-normalized net cone resistance
        rvcresistance_n:  float | representative value of net cone
            resistance (MPa)
        rvcresistance_osnn:  float | representative value of overburden
            stress-normalized net cone resistance
        rvsresistance:  float | representative value of sleeve resistance (kPa)
        rvdeltaq:  float | representative value of DeltaQ
        rvfratio:  float | rrepresentative value of friction ratio
        sid:  str | soil ID of the soil associated with the test
        sparameter:  float | state parameter (decimal)
        rdensity:  float | relative density (decimal)
        vratio:  float | void ratio (decimal)
        oratio:  float | overconsolidation ratio
        hid:  str | calibration chamber ID
        hdiameter:  float | chamber diameter (cm)
        hheight:  float | chamber height (cm)
        bcondition:  int | boundary conditions (see table below)
            -----------------------------------------------------------
            | Boundary Condition | Side Restraint   | Base Restraint  |
            -----------------------------------------------------------
            | 1                  | Constant Stress  | Constant Stress |
            | 2                  | Zero Strain      | Zero Strain     |
            | 3                  | Zero Strain      | Constant Stress |
            | 4                  | Constant Stress  | Zero Strain     |
            | 5                  | Servo-controlled | Constant Stress |
            | -999999            | Not Assigned     | Not Assigned    |
            -----------------------------------------------------------
        htvstress:  float | chamber total vertical stress (kPa)
        hthstress:  float | chamber total horizontal stress (kPa)
        hevstress:  float | chamber effective vertical stress (kPa)
        hehstress:  float | chamber effective horizontal stress (kPa)
        htmstress:  float | chamber mean total stress (kPa)
        hemstress:  float | chamber mean effective stress (kPa)
        saturation:  bool | True if the soil is saturated; False otherwise
        hppressure:  float | calibration chamber pore pressure
        ccolor:  str | test color (for plotting purposes)
        cmarker:  str | Pyplot marker style (see table below)
            ------------------------
            | 'o' | circle         |
            | 's' | square         |
            | 'D' | diamond        |
            | 'p' | plus sign      |
            | '*' | star           |
            | 'P' | pentagon       |
            | 'X' | filled x       |
            | '^' | triangle up    |
            | 'v' | triangle down  |
            | '<' | triangle left  |
            | '>' | triangle right |
            ------------------------

        """
        # Get the connection and cptc sqlalchemy objects.
        connection, table = get_table()

        # Filter the Soils Database.
        stmt = select([table])

        # Filter by Calibration Chamber CPT ID.
        if cid:
            stmt = stmt.where(table.columns.cid.in_((cid)))

        # Filter by soil ID.
        if sid:
            stmt = stmt.where(table.columns.sid.in_((sid)))

        # Filter by boundary condition.
        if bcondition:
            stmt = stmt.where(table.columns.bcondition.in_(bcondition))

        # Filter by cone diameter.
        if cdiameter:
            stmt = stmt.where(
                between(
                    table.columns.cdiameter,
                    cdiameter[0],
                    cdiameter[1]
                )
            )

        # Filter by whether or not cone resistance readings are attached.
        if cresistance:
            stmt = stmt.where(table.columns.cresistance.in_(cresistance))

        # Filter by whether or not sleeve resistance reading are attached.
        if sresistance:
            stmt = stmt.where(table.columns.sresistance.in_(sresistance))

        # Filter by whether or not cone horitzontal pressure readings are
        # attached.
        if chpressure:
            stmt = stmt.where(table.columns.chpressure.in_(chpressure))

        # Filter by whether or not cone pore pressure readings are attached.
        if cppressure:
            stmt = stmt.where(table.columns.cppressure.in_(cppressure))

        # Filter by chamber total horizontal stress.
        if hthstress:
            stmt = stmt.where(
                between(
                    table.columns.hthstress,
                    hthstress[0],
                    hthstress[1]
                )
            )

        # Filter by chamber total vertical stress.
        if htvstress:
            stmt = stmt.where(
                between(
                    table.columns.htvstress,
                    htvstress[0],
                    htvstress[1]
                )
            )

        # Filter by chamber mean stress.
        if htmstress:
            stmt = stmt.where(
                between(
                    table.columns.htmstress,
                    htmstress[0],
                    htmstress[1]
                )
            )

        if hemstress:
            stmt = stmt.where(
                between(
                    table.columns.hemstress,
                    hemstress[0],
                    hemstress[1]
                )
            )

        # Filter by chamber pore pressure.
        if hppressure:
            stmt = stmt.where(
                between(
                    table.columns.hppressure,
                    hppressure[0],
                    hppressure[1]
                )
            )

        # Filter by overconsolidation ratio.
        if oratio:
            stmt = stmt.where(
                between(
                    table.columns.oratio,
                    oratio[0],
                    oratio[1]
                )
            )

        # Filter by cone resistance.
        if rvcresistance:
            stmt = stmt.where(
                between(
                    table.columns.rvcresistance,
                    rvcresistance[0],
                    rvcresistance[1]
                )
            )

        # Filter by mean effective stress normalized net cone resistance.
        if rvcresistance_mesnn:
            stmt = stmt.where(
                between(
                    table.columns.rvcresistance_mesnn,
                    rvcresistance_mesnn[0],
                    rvcresistance_mesnn[1]
                )
            )

        # Filter by net cone resistance.
        if rvcresistance_n:
            stmt = stmt.where(
                between(
                    table.columns.rvcresistance_n,
                    rvcresistance_n[0],
                    rvcresistance_n[1]
                )
            )

        # Filter by overburden stress normalized net cone resistance.
        if rvcresistance_osnn:
            stmt = stmt.where(
                between(
                    table.columns.rvcresistance_osnn,
                    rvcresistance_osnn[0],
                    rvcresistance_osnn[1]
                )
            )

        # Filter by relative density.
        if rdensity:
            stmt = stmt.where(
                between(
                    table.columns.rdensity,
                    rdensity[0],
                    rdensity[1]
                )
            )

        # Filter by sleeve resistance.
        if rvsresistance:
            stmt = stmt.where(
                between(
                    table.columns.rvsresistance,
                    rvsresistance[0],
                    rvsresistance[1]
                )
            )

        # Filter by Delta Q.
        if rvdeltaq:
            stmt = stmt.where(
                between(
                    table.columns.rvdeltaq,
                    rvdeltaq[0],
                    rvdeltaq[1]
                )
            )

        # Filter by friction ratio.
        if rvfratio:
            stmt = stmt.where(
                between(
                    table.columns.rvfratio,
                    rvfratio[0],
                    rvfratio[1]
                )
            )

        # Filter by saturation.
        if saturation:
            stmt = stmt.where(table.columns.saturation.in_(saturation))

        # Filter by state parameter.
        if sparameter:
            stmt = stmt.where(
                between(
                    table.columns.sparameter,
                    sparameter[0],
                    sparameter[1]
                )
            )

        # Filter by void ratio.
        if vratio:
            stmt = stmt.where(
                between(
                    table.columns.vratio,
                    vratio[0],
                    vratio[1]
                )
            )

        # Make the dataframe.
        self.__results = pd.read_sql(
            stmt,
            connection
        ).set_index('cid')

    @property
    def results(self):
        return self.__results


def create_cptc(
    cname,
    crname,
    crdate,
    sid,
    hid,
    cdiameter=None,
    rvcresistance=None,
    rvsresistance=None,
    vratio=None,
    oratio=None,
    bcondition=None,
    htvstress=None,
    hthstress=None,
    saturation=None,
    hppressure=None,
    crdata=False,
    srdata=False,
    chpdata=False,
    cppdata=False,
    ccolor=None,
    cmarker=None
):
    """
    Create a record in the Calibration Chamber CPT database.

    Parameters
    ----------
    crname:  str | name of the reference where the test came
    crdate:  float | date the reference was published
    sid:  str | soil ID
    hid:  str | calibration chamber ID
    cdiameter:  float | cone diameter (cm)
    rvcresistance:  float | representative value of cone resistance (MPa) for
        the test
    rvsresistance:  float | representative value of sleeve resistance (kPa)
    rvdeltaq:  float | representative value of DeltaQ for the test
    sparameter:  float | state parameter (decimal)
    rdensity:  float | relative density (decimal)
    vratio:  float | void ratio (decimal)
    oratio:  float | overconsolidation ratio
    dratio:  float | diameter ratio (chamber diameter/cone diameter)
    hdiameter:  float | chamber diameter (cm)
    hheight:  float | chamber height (cm)
    bcondition:  int | boundary conditions (see table below)
        -----------------------------------------------------------
        | Boundary Condition | Side Restraint   | Base Restraint  |
        -----------------------------------------------------------
        | 1                  | Constant Stress  | Constant Stress |
        | 2                  | Zero Strain      | Zero Strain     |
        | 3                  | Zero Strain      | Constant Stress |
        | 4                  | Constant Stress  | Zero Strain     |
        | 5                  | Servo-controlled | Constant Stress |
        | -999999            | Not Assigned     | Not Assigned    |
        -----------------------------------------------------------
    htvstress:  float | chamber total vertical stress (kPa)
    hthstress:  float | chamber total horizontal stress (kPa)
    saturation:  bool | True if the soil is saturated; False otherwise
    hppressure:  float | calibration chamber pore pressure
    hevstress:  float | chamber effective vertical stress (kPa)
    hehstress:  float | chamber effective horizontal stress (kPa)
    crdata:  list |  cone resistance data in the form
        [cone resistance (MPa), depth (cm)]
    srdata:  list | sleeve resistance data in the form
        [sleeve resistance (kPa), depth (cm)]
    chpdata:  list |  cone horizontal data in the form
        [cone horizontal pressure (kPa), depth (cm)]
    cppdata:  list |  cone pore presssure data in the form
        [cone pore pressure (kPa), depth (cm)]
    ccolor:  str | test color (for plotting purposes)
    cmarker:  str | Pyplot marker style (see table below)
        ------------------------
        | 'o' | circle         |
        | 's' | square         |
        | 'D' | diamond        |
        | 'p' | plus sign      |
        | '*' | star           |
        | 'P' | pentagon       |
        | 'X' | filled x       |
        | '^' | triangle up    |
        | 'v' | triangle down  |
        | '<' | triangle left  |
        | '>' | triangle right |
        ------------------------

    """
    # Remove special characters, spaces, etc. in sname and srname.
    part1 = ''.join(char for char in str(cname) if char.isalnum()).lower()
    part2 = ''.join(char for char in crname if char.isalnum()).lower()[:3]

    # Make the soil ID.
    cid = part1 + '-' + part2 + '-' + str(crdate)[2:]

    # Get the calibration chamber data.
    htable = chamber.Chamber(hid=[hid])
    hdiameter = htable.results.loc[hid, 'hdiameter']
    hheight = htable.results.loc[hid, 'hheight']

    # If cone data is present, calculate the representative values.
    if crdata:

        # Set cresistance (the parameter that indicates if cone resistance
        # data is attached) at True.
        cresistance = True

        # Calculate the representative value of cone resistance.
        rvcresistance = rvalue(
            cdiameter,
            hdiameter,
            hheight,
            crdata[0],
            crdata[1]
        )

        # Pickle the data.
        with open(
            'cresistance/{}.p'.format(cid),
            'wb'
        ) as out_file:
            pickle.dump(
                crdata,
                out_file
            )

    else:

        # Set cresistance (the parameter that indicates if cone resistance
        # data is attached) at False.
        cresistance = False

    if srdata:

        # Set cresistance (the parameter that indicates if sleeve resistance
        # data is attached) at True.
        sresistance = True

        # Calculate the representative value of sleeve resistance.
        rvsresistance = rvalue(
            cdiameter,
            hdiameter,
            hheight,
            srdata[0],
            srdata[1]
        )

        # Pickle the data.
        with open(
            'sresistance/{}.p'.format(cid),
            'wb'
        ) as out_file:
            pickle.dump(
                srdata,
                out_file
            )

    else:

        # Set cresistance (the parameter that indicates if sleeve resistance
        # data is attached) at False.
        sresistance = False

    if chpdata:

        # Set cresistance (the parameter that indicates if cone horizontal
        # resistance data is attached) to True.
        chpressure = True

        # Calculate the representative value of cone resistance.
        rvchpressure = rvalue(
            cdiameter,
            hdiameter,
            hheight,
            chpdata[0],
            chpdata[1]
        )

        # Pickle the data.
        with open(
            'chpressure/{}.p'.format(cid),
            'wb'
        ) as out_file:
            pickle.dump(
                chpdata,
                out_file
            )

    else:

        # Set cresistance (the parameter that indicates if cone resistance
        # data is attached) at False.
        chpressure = False

    if cppdata:

        # Set cresistance (the parameter that indicates if cone horizontal
        # resistance data is attached) at True.
        cppresure = True

        # Calculate the representative value of cone resistance.
        #rvchppressure = rvalue(
        #    cdiameter,
        #    hdiameter,
        #    hheight,
        #    cresistances[0],
        #    cresistances[1]
        #)

        # Pickle the data.
        #with open(
        #    'cresistance/{}.p'.format(cid),
        #    'wb'
        #) as out_file:
        #    pickle.dump(
        #        cresistances,
        #        out_file
        #    )

    else:

        # Set cresistance (the parameter that indicates if cone resistance
        # data is attached) at False.
        cppressure = False

    # If saturated and if the chamber pore pressure has been input, calculate
    # chamber effective vertical and horizontal stresses.
    if (
        saturation and
        hppressure
    ):
        hevstress = htvstress - hppressure
        hehstress = hthstress - hppressure

    elif (
        saturation and
        not hppressure
    ):
        raise ValueError(
            'If the test is saturated, the chamber pore pressure must be '
            'greater than 0.')

    elif (
        not saturation and
        not hppressure
    ):

        hevstress = htvstress
        hehstress = hthstress

    elif (
        not saturation and
        hppressure
    ):
        raise ValueError(
            'If the test is not saturated, the chamber pore pressure must be '
            '0.'
        )

    # Chamber mean stress
    htmstress = calc_htmstress(htvstress, hthstress)
    hemstress = calc_hemstress(hevstress, hehstress)

    # Mean effective stress-normalized net cone resistance.
    rvcresistance_mesnn = calc_rvcresistance_mesnn(
        rvcresistance,
        hemstress,
        htmstress
    )

    # If there is a representative value of sleeve resistance, calculate
    # the represntative value of effective stress-normalized sleeve resistance,
    # overburden stress-normalized net cone resistance, and the representative
    # value of Delta Q.
    if (
        rvcresistance and
        rvsresistance
    ):

        rvcresistance_osnn = calc_rvcresistance_osnn(
            rvcresistance,
            htvstress,
            hevstress
        )

        rvsresistance_esn = calc_rvsresistance_esn(
            rvsresistance,
            hevstress
        )

        rvdeltaq = calc_rvdeltaq(
            rvcresistance_osnn,
            rvsresistance_esn
        )

    else:
        rvcresistance_osnn = None
        rvsresistance_esn = None
        rvdeltaq = None

    # If there is a representative value of cone resistance and chamber total
    # effective vertical stress, calculate net cone resistance.
    if (
        rvcresistance and
        htvstress
    ):
        rvcresistance_n = calc_rvcresistance_n(rvcresistance, htvstress)

    else:
        rvcresistance_n = None

    # If there is aa representative value of net cone and sleeve resistance,
    # calculate friction ratio.
    if (
        rvcresistance_n and
        rvsresistance
    ):
        rvfratio = calc_rvfratio(
            rvcresistance_n,
            rvsresistance
        )

    else:
        rvfratio = None

    # If the void ratio is given, calculate relative density and state
    # parameter.
    if vratio:

        # Get soil data.
        stable = soil.Soil(sid=[sid])
        compressibility = stable.results.at[sid, 'compressibility']
        intercept = stable.results.at[sid, 'intercept']
        voidx = stable.results.at[sid, 'voidx']
        voidn = stable.results.at[sid, 'voidn']

        # Calculate relative density.
        rdensity = calc_rdensity(
            voidx,
            voidn,
            vratio
        )

        # Calculate state parameter.
        sparameter = calc_sparameter(
            compressibility,
            intercept,
            hehstress,
            vratio
        )

    else:
        rdensity = None
        sparameter = None

    # If the cone diameter is given, calculate diameter ratio.
    if (
        cdiameter and
        hdiameter
    ):
        dratio = hdiameter / cdiameter

    elif (
        not cdiameter or
        not hdiameter
    ):
        warnings.warn(
            'Cone or calibration chamber diameter is invalid.  Diameter ratio '
            'was not calculated.'
        )

        dratio = None

    # Create a dictionary containing the columns and values.
    values = {
        'cid': cid,
        'cname': cname,
        'crname': crname,
        'crdate': crdate,
        'cdiameter': cdiameter,
        'cresistance': cresistance,
        'sresistance': sresistance,
        'cppressure': cppressure,
        'chpressure': chpressure,
        'rvcresistance': rvcresistance,
        'rvcresistance_mesnn': rvcresistance_mesnn,
        'rvcresistance_n': rvcresistance_n,
        'rvcresistance_osnn': rvcresistance_osnn,
        'rvsresistance': rvsresistance,
        'rvsresistance_esn': rvsresistance_esn,
        'rvdeltaq': rvdeltaq,
        'rvfratio': rvfratio,
        'sid': sid,
        'vratio': vratio,
        'sparameter': sparameter,
        'rdensity': rdensity,
        'oratio': oratio,
        'hid': hid,
        'dratio': dratio,
        'bcondition': bcondition,
        'htvstress': htvstress,
        'hthstress': hthstress,
        'saturation': saturation,
        'hppressure': hppressure,
        'hevstress': hevstress,
        'hehstress': hehstress,
        'htmstress': htmstress,
        'hemstress': hemstress,
        'ccolor': ccolor,
        'cmarker': cmarker
    }

    # Get the connection and the soils table objects.
    connection, cptc = get_table()

    # Update the database.
    connection.execute(insert(cptc), [values])


def get_table():
    """
    Get the CPTc table.

    Returns
    -------
    connection:  sqlalchemy connection object | connection to the Calibration
        Chamber CPT database
    table:  sqlalchemy Table object | Calibation Chamber CPT database

    """
    # Make the engine to connect to the database.
    engine = create_engine('sqlite:///db/geotechnipy.sqlite')

    # Make the connection object.
    connection = engine.connect()

    # Make the metadata object.
    metadata = MetaData()

    # Reflect the table.
    table = Table('cptc', metadata, autoload=True, autoload_with=engine)

    return connection, table


def rdensity2vratio(
    voidx,
    voidn,
    rdensity
):
    """
    Calculate void ratio from relative density.

    This function is intended only to be used when creating a soil record.

    Parameters
    ----------
    rdensity:  float | relative density (decimal)
    voidn:  float | minimum void ratio (decimal)
    voidx:  float | maximum void ratio (decimal)

    Returns
    -------
    vratio:  float | void ratio

    """
    vratio = voidx - rdensity * (voidx - voidn)

    return vratio


def rvalue(
    cdiameter,
    hdiameter,
    hheight,
    readings,
    depths
):
    """
    Calculate the representative value for calibration chamber CPT data.

    This function calculates the representative value of either the cone
    resistance, sleeve resistance, or Delta Q of a calibration chamber CPT
    test.  This function is based on an analysis of calibration chamber CPT
    data presented in Gamez (2020) where it was found that the data reaches
    an ultimate value between a depth of 4 times the cone diameter below
    the top of the calibration chamber and a height of and a height of 8
    times the cone diameter above the bottom of the calibration chamber.

    Parameters
    ----------
    cdiameter:  float | cone diameter (cm)
    hdiameter:  float | calibration chamber diameter (cm)
    hheight:  float | calibration chamber diameter (cm)
    readings:  tuple | CPT readings (e.g., cone resistance, sleeve resistance)
    depths:  tuple | depths at which the readings were performed (cm)

    Returns
    -------
    rval:  float | representative value of the calibration chamber CPT

    """
    readings = np.array(readings)
    depths = np.array(depths)

    # Calculate the depth of the upper boundary
    upper = 4 * cdiameter

    # Calculate the depth of the lower boundary
    lower = hheight - 8 * cdiameter

    # Find the indices of the readings that are below the upper boundary and
    # above the lower boundary.
    # Note:  The where function returns an array with row indices and column
    # indices.  For a 1d array, which is what the where is being applied to
    # (i.e., data[0]), the numpy array is represented as a single columns with
    # many rows, rather than a single row with many columns.  Thus, the zero
    # index at the end of the where function returns the indices of the rows
    # rather than columns as might be expected.
    indices = np.where(
        np.logical_and(
            depths >= upper,
            depths <= lower
        )
    )[0]

    # Calculate the representative value.
    rval = np.mean(readings[indices])

    return rval


def update_table(
    cid,
    **kwargs
):
    """
    Updates 1 soil record in the Soils database.

    Parameters
    ----------
    cid:  str | calibration chamber CPT ID

    Other Parameters
    ----------------
    crname:  str | name of the reference where the test came
    crdate:  float | date the reference was published
    cdiameter:  float | cone diameter (cm)
    cresistance:  bool | True if cone resistance (MPa) versus depth attached;
        False otherwise
    sresistance:  bool | True if sleeve resistance (kPa) versus depth attached;
        False otherwise
    cppressure:  bool | True if CPT pore pressure (kPa) versus depth attached;
        False otherwise
    chpressure:  bool | True if CPT lateral stress (kPa) versus depth attached;
        False otherwise
    rvcresistance:  float | representative value of cone resistance (MPa) for
        the test
    rvsresistance:  float | representative value of sleeve resistance (kPa)
    rvdeltaq:  float | representative value of DeltaQ for the test
    sid:  str | soil ID of the soil associated with the test
    sparameter:  float | state parameter (decimal)
    rdensity:  float | relative density (decimal)
    vratio:  float | void ratio (decimal)
    oratio:  float | overconsolidation ratio
    hdiameter:  float | chamber diameter (cm)
    hheight:  float | chamber height (cm)
    bcondition:  int | boundary conditions (see table below)
        -----------------------------------------------------------
        | Boundary Condition | Side Restraint   | Base Restraint  |
        -----------------------------------------------------------
        | 1                  | Constant Stress  | Constant Stress |
        | 2                  | Zero Strain      | Zero Strain     |
        | 3                  | Zero Strain      | Constant Stress |
        | 4                  | Constant Stress  | Zero Strain     |
        | 5                  | Servo-controlled | Constant Stress |
        | -999999            | Not Assigned     | Not Assigned    |
        -----------------------------------------------------------
    htvstress:  float | chamber total vertical stress (kPa)
    hthstress:  float | chamber total horizontal stress (kPa)
    saturation:  bool | True if the soil is saturated; False otherwise
    hppressure:  float | calibration chamber pore pressure
    ccolor:  str | test color (for plotting purposes)
    cmarker:  str | Pyplot marker style (see table below)
        ------------------------
        | 'o' | circle         |
        | 's' | square         |
        | 'D' | diamond        |
        | 'p' | plus sign      |
        | '*' | star           |
        | 'P' | pentagon       |
        | 'X' | filled x       |
        | '^' | triangle up    |
        | 'v' | triangle down  |
        | '<' | triangle left  |
        | '>' | triangle right |
        ------------------------

    """
    # Get the connection and soils table objects.
    connection, table = get_table()

    # Create the query statement.
    stmt = update(table)
    stmt = stmt.where(table.columns.cid==cid)

    # Update the database.
    connection.execute(stmt, [kwargs])


def create_cresistance(
    cid,
    readings,
    depths

):
    """
    Create a file containing cone resistance vs depth.

    Parameters
    ----------
    cid:  str | Calibration chamber CPT ID
    readings:  list | readings of cone resistance (MPa)
    depths:  list | depths at which readings were taken (cm)
    update:  bool | True to update rvcresistance; False otherwise

    """
    # Combine the sieve analysis data.
    data = (readings, depths)

    # Pickle the sieve analysis data.
    with open(
        'cresistance/{}.p'.format(cid),
        'wb'
    ) as out_file:
        pickle.dump(
            data,
            out_file
        )

    update_table(
        cid,
        cresistance=True
    )


def create_sresistance(
    cid,
    readings,
    depths
):
    """
    Create a file containing cone resistance vs depth.

    Parameters
    ----------
    cid:  str | Calibration chamber CPT ID
    readings:  list | readings of sleeve resistance (kPa)
    depths:  list | depths at which readings were taken (cm)
    update:  bool | True to update rvsresistance; False otherwise

    """
    # Combine the sieve analysis data.
    data = (readings, depths)

    # Pickle the sieve analysis data.
    with open(
        'sresistance/{}.p'.format(cid),
        'wb'
    ) as out_file:
        pickle.dump(
            data,
            out_file
        )

    update_table(
        cid,
        sresistance=True
    )


def create_cppressure(
    cid,
    readings,
    depths,
):
    """
    Create a file containing penetration-induced pore pressure vs depth.

    Parameters
    ----------
    cid:  str | Calibration chamber CPT ID
    readings:  list | readings of penetration induced pore pressure (kPa)
    depths:  list | depths at which readings were taken (cm)

    """
    # Combine the data.
    data = (readings, depths)

    # Pickle the data.
    with open(
        'cppressure/{}.p'.format(cid),
        'wb'
    ) as out_file:
        pickle.dump(
            data,
            out_file
        )

    update_table(
        cid,
        cppressure=True
    )

def create_chpressure(
    cid,
    readings,
    depths
):
    """
    Create a file containing penetration-induced horizontal pressure vs depth.

    Parameters
    ----------
    cid:  str | Calibration chamber CPT ID
    readings:  list | readings of penetration-induced horizontal pressure (kPa)
    depths:  list | depths at which readings were taken (cm)

    """
    # Combine the sieve analysis data.
    data = (readings, depths)

    # Pickle the sieve analysis data.
    with open(
        'chpressure/{}.p'.format(cid),
        'wb'
    ) as out_file:
        pickle.dump(
            readings,
            out_file
        )

    update_table(
        cid,
        chpressure=True
    )


def calc_hemstress(
    hevstress,
    hehstress
):
    """
    Calculate the chamber mean effective stress.

    Parameters
    ----------
    hevstress:  float | chamber effective vertical stress (kPa)
    hehstress:  float | chamber effective horizontal stress (kPa)

    Returns
    -------
    hemstress:  float | chamber mean effective stress (kPa)

    """
    hemstress = (hevstress + hehstress * 2) / 3

    return hemstress


def calc_htmstress(
    htvstress,
    hthstress
):
    """
    Calculate the chamber mean total stress.

    Parameters
    ----------
    hevstress:  float | chamber total vertical stress (kPa)
    hehstress:  float | chamber total horizontal stress (kPa)

    Returns
    -------
    hemstress:  float | chamber mean total stress (kPa)

    """
    htmstress = (htvstress + hthstress * 2) / 3

    return htmstress


def calc_rvcresistance_mesnn(
    rvcresistance,
    hemstress,
    htmstress
):
    """
    Calculate mean-effective-stress-normalized net cone resistance.

    The mean-effective-stress-normalized cone resistance, [q_c]_P', is
    defined by the equation

    [q_c]_P' = (q_c - P) / P'

    where

    [q_c]_P'    = tip Resistance
    P           = (sigma_1 + sigma_3 * 2) / 3
    P'          = (sigma_1' + sigma_3' * 2) / 3

    Parameters
    ----------
    rvcresistance:  float | representative value of cone resistance for the
        CPT calibration chamber test (MPa)
    hemstress:  float | calibration chamber mean effective stress (kPa)
    htmstress:  float | calibration chamber mean total stress (kPa)

    Returns
    -------
    rvcresistance_mesnn:  float | mean effective stress normalized cone
        resistance

    """
    # Convert from MPa to kPa.
    rvcresistance = rvcresistance * 1000

    rvcresistance_mesnn = (rvcresistance - htmstress) / hemstress

    return rvcresistance_mesnn


def calc_rvcresistance_osnn(
    rvcresistance,
    htvstress,
    hevstress
):
    """
    Calculate overburden stress-normalized net corrected cone resistance.

    Parameters
    ----------
    rvcresistance:  float | representative value of cone resistance for a
        CPT calibration chamber test (MPa)
    htvstress:  float | chamber total vertical stress (kPa)
    hevstress:  float | chamber effective vertical stres (kPa)

    Returns
    -------
    rvcresistance_osnn:  float | representative value of overburden stress-
        normalized net cone resistance for a CPT calibration chamber
        test (no units)

    """
    # Note:  the cone resistance must be convert from MPa to kPa.
    rvcresistance_osnn = ((rvcresistance * 1000) - htvstress) / hevstress

    return rvcresistance_osnn


def calc_rdensity(
    voidx,
    voidn,
    vratio
):
    """
    Calculate relative density.
    """
    if (
        voidx and
        voidn
    ):
        rdensity = (voidx - vratio) / (voidx - voidn)

    elif (
        not voidx or
        not voidn
    ):
        warnings.warn(
            'Maximum or minimum void ratio is invalid. Relative density '
            'was not calculated.'
        )

        rdensity = None

    return rdensity


def calc_rvcresistance_n(
    rvcresistance,
    htvstress
):
    """
    Calculate the representative value of net cone resistance.

    """
    rvcresistance_n = rvcresistance - (htvstress / 1000)

    return rvcresistance_n


def calc_rvdeltaq(
    rvcresistance_osnn,
    rvsresistance_esn
):
    """
    Calculate Delta Q.

    Parameters
    ----------
    rvcresistance_osnn:  float | representative value of overburden stress-
        normalized net corrected tip resistance for a CPT calibration chamber
        test
    rvsresistance_esn:  float | representative value of effective stress-
        normalized sleeve resistance for a CPT calibration chamber test

    Returns
    -------
    rvdeltaq:  float | representative value of Delta Q for a CPT calibration
        chamber test

    """
    rvdeltaq = (rvcresistance_osnn + 10) / (rvsresistance_esn + 0.67)

    return rvdeltaq


def calc_rvfratio(
    rvcresistance_n,
    rvsresistance
):
    """
    Calculate the friction ratio.

    Parameters
    ----------
    rvcresistance_n:  float | representative value of net cone resistance (MPa)
    rvsresistance:  float | representative value of sleeve resistance (kPa)

    Returns
    -------
    rvfratio:  float | representative value of friction ratio (decimal)

    """
    rvfratio = (rvsresistance / 1000) / rvcresistance_n

    return rvfratio


def calc_rvsresistance_esn(
    rvsresistance,
    hevstress
):
    """
    Calculate effective stress-normalized sleeve resistance.

    Parameters
    ----------
    rvsresistance:  float | representative value of sleeve resistance for a
        CPT calibration chamber tests (kPa)
    hevstress:  float | chamber effective vertical stress (kPa)

    Returns
    -------
    rvsresistance_esn:  float | representative value of effective stress-
        normalized sleeve resistance for a CPT calibration chamber test

    """
    rvsresistance_esn = rvsresistance / hevstress

    return rvsresistance_esn


def calc_sparameter(
    compressibility,
    intercept,
    hehstress,
    vratio
):
    """
    Calculate the state parameter.

    """
    if (
        compressibility and
        intercept and
        hehstress
    ):
        sparameter = vratio - (
            compressibility * np.log10(hehstress) + intercept
        )

    elif (
        not compressibility or
        not intercept or
        not hehstress
    ):
        warnings.warn(
            'Compressibility, intercept, or chamber effective horizontal '
            'stress is invalid. State parameter was not calculated.'
        )

        sparameter = None

    return sparameter
