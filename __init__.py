#from .chamber import chamber
#from .cpt import cpt
#from .cptc import cptc
#from .dcp import dcp
#from .soil import soil
import os
from sqlalchemy import (
    Boolean,
    Column,
    create_engine,
    Float,
    Integer,
    MetaData,
    select,
    String,
    Table,
    update
)


def reset_chamber():
    """
    Reset the Calibration Chamber database.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    table = Table(
        'chamber',
        metadata,
        autoload=True,
        autoload_with=engine
    )

    engine.execute(table.delete())


def reset_cptc():
    """
    Reset the Calibration Chamber CPT database.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    table = Table(
        'cptc',
        metadata,
        autoload=True,
        autoload_with=engine
    )

    engine.execute(table.delete())


def reset_dcp():
    """
    Reset the DCP database.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    table = Table(
        'dcp',
        metadata,
        autoload=True,
        autoload_with=engine
    )

    engine.execute(table.delete())


def reset_soil():
    """
    Reset the Soil database.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    table = Table(
        'soil',
        metadata,
        autoload=True,
        autoload_with=engine
    )

    engine.execute(table.delete())


if os.path.isdir('db') is False:
    os.makedirs('db')

if os.path.isdir('figs') is False:
    os.makedirs('figs')

# Calibration Chamber CPT
if os.path.isdir('cresistance') is False:
    os.makedirs('cresistance')

if os.path.isdir('sresistance') is False:
    os.makedirs('sresistance')

if os.path.isdir('cppressure') is False:
    os.makedirs('cppressure')

if os.path.isdir('chpressure') is False:
    os.makedirs('chpressure')

# Dynamic Cone Penetrometer
if os.path.isdir('bcounts') is False:
    os.makedirs('bcounts')

# Soil
if os.path.isdir('gradations') is False:
    os.makedirs('gradations')

# Make the engine to connect to the database.
engine = create_engine('sqlite:///db/geotechnipy.sqlite')

# Initialize the metadata.
metadata = MetaData()

if os.path.isfile('db/geotechnipy.sqlite') is False:

    """
    Make the Calibration Chamber CPT table.

    Parameters
    ----------
    cid:  str | calibration chamber CPT ID
    crname:  str | name of the reference where the test came
    crdate:  float | date the reference was published
    cdiameter:  float | cone diameter
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
    rvcresistance_mesnn:  float | representative value of mean effective
        stress-normalized net cone resistance
    rvcresistance_n:  float | representative value of net cone resistance (MPa)
    rvcresistance_osnn:  float | representative of cone overburden stress-
        normalized net corrected tip resistance
    rvsresistance:  float | representative value of sleeve resistance (kPa)
    rvsresistance_esn:  float representative value of effective stress
        normalized sleeve resistance
    rvdeltaq:  float | representative value of DeltaQ for the test
    rvfratio:  float | representative value of friction ratio
    sid:  str | soil ID of the soil associated with the test
    sparameter:  float | state parameter (decimal)
    rdensity:  float | relative density (decimal)
    vratio:  float | void ratio (decimal)
    oratio:  float | overconsolidation ratio
    hid:  str | calibration chamber ID
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
    hppressure:  float | calibration chamber pore pressure (kPa)
    hevstress:  float | chamber effective vertical stress (kPa)
    hehstress:  float | chamber effective horizontal stress (kPa)
    htmstress:  float | chamber mean total stress (kPa)
    hemstress:  float | chamber mean effective stress (kPa)
    ccolor:  str | test color (for plotting purposes)
    cmarker:  str | Pyplot marker style

    """
    Table(
        'cptc',
        metadata,
        Column('cid', String(), unique=True),
        Column('cname', String()),
        Column('crname', String()),
        Column('crdate', Integer()),
        Column('cdiameter', Float()),
        Column('cresistance', Boolean()),
        Column('sresistance', Boolean()),
        Column('cppressure', Boolean()),
        Column('chpressure', Boolean()),
        Column('rvcresistance', Float()),
        Column('rvcresistance_mesnn', Float()),
        Column('rvcresistance_n', Float()),
        Column('rvcresistance_osnn', Float()),
        Column('rvsresistance', Float()),
        Column('rvsresistance_esn', Float()),
        Column('rvchpressure', Float()),
        Column('rvdeltaq', Float()),
        Column('rvfratio', Float()),
        Column('sid', String()),
        Column('vratio', Float()),
        Column('sparameter', Float()),
        Column('rdensity', Float()),
        Column('oratio', Float()),
        Column('hid', String()),
        Column('dratio', Float()),
        Column('bcondition', Integer()),
        Column('htvstress', Float()),
        Column('hthstress', Float()),
        Column('saturation', Boolean()),
        Column('hppressure', Float()),
        Column('hevstress', Float()),
        Column('hehstress', Float()),
        Column('htmstress', Float()),
        Column('hemstress', Float()),
        Column('ccolor', String(), default='black'),
        Column('cmarker', String(), default='o')
    )

    """
    Make the Calibration Chamber database.

    Parameters
    ----------
    hid:  str | calibration chamber ID
    hname:  str | calibration chamber name
    hrname:  str | name of the reference where the test came
    hrdate:  float | date the reference was publishe
    hdiameter:  float | chamber diameter (cm)
    hheight:  float | chamber height (cm)

    """
    Table(
        'chamber',
        metadata,
        Column('hid', String, primary_key=True, unique=True),
        Column('hname', String),
        Column('hrname', String),
        Column('hrdate', Integer),
        Column('hdiameter', Float),
        Column('hheight', Float)
    )

    """
    Create the DCP database.

    Other Parameters
    ----------------
    did:  str | DCP ID
    drname:  str | name of the reference where the test came
    drdate:  float | date the reference was published
    bcount:  bool | False if blow counts are not attached; True otherwise
    dmass:  float | mass of the hammer (kg)
    dcolor:  str | test color (for plotting purposes)
    dmarker:  str | Pyplot marker style (see table below)
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
    Table(
        'dcp',
        metadata,
        Column('did', String(), primary_key=True, unique=True),
        Column('dname', String()),
        Column('drname', String()),
        Column('drdate', Integer()),
        Column('bcount', Boolean()),
        Column('dmass', Float()),
        Column('dcolor', String(), default='black'),
        Column('dmarker', String(), default='o')
    )

    """
    Create the Soils database.

    Parameters
    ----------
    sid:  str | soil ID
    sname:  str | soil name (e.g., Ottawa sand)
    srname:  str | name of the reference where the soil came from
    srdate:  float | date the reference was published
    compressibility:  float | slope of the Critical State Line
    intercept:  float | intercept of the Critical State Line
    deltaq:  float | delta Q (see Saye et al. 2017)
    voidx:  float | maximum void ratio
    voidn:  float | minimum void ratio
    llimit:  float | liquid limit
    odllimit:  float | oven dried liquid limit
    pindex:  float | plasticity index
    plimit:  float | plastic limit
    gradation:  Bool | indicates if sieve analysis results are attached
    gravel:  float | gravel content (decimal)
    sand:  float | sand content (decimal)
    fines:  float | fines content (decimal)
    diameter10:  float | particle diameter at 10% passing
    diameter30:  float | particle diameter at 30% passing
    diameter50:  float | particle diameter at 50% passing
    diameter60:  float | particle diameter at 60% passing
    curvature:  float | coefficient of curvature
    uniformity:  float | coefficient of uniformity
    organic:  Bool | True if the soil is organic (e.g., peat), False otherwise
    roundness:  float | particle roundness
    scolor:  str | soil color (for plotting purposes)
    sgravity:  float | specific gravity
    shape:  str | particle shape
    smarker:  str | Pyplot marker style
    sphericity:  float | particle sphericity
    uscs:  str | USCS classification

    """
    Table(
        'soil',
        metadata,
        Column('sid', String(), primary_key=True, unique=True),
        Column('sname', String()),
        Column('srname', String()),
        Column('srdate', Integer()),
        Column('compressibility', Float()),
        Column('intercept', Float()),
        Column('deltaq', Float()),
        Column('voidx', Float()),
        Column('voidn', Float()),
        Column('llimit', Float()),
        Column('odllimit', Float()),
        Column('plimit', Float()),
        Column('pindex', Float()),
        Column('sgravity', Float()),
        Column('uscs', Integer()),
        Column('gradation', Boolean()),
        Column('fines', Float()),
        Column('sand', Float()),
        Column('diameter10', Float()),
        Column('diameter30', Float()),
        Column('diameter50', Float()),
        Column('diameter60', Float()),
        Column('curvature', Float()),
        Column('uniformity', Float()),
        Column('shape', Integer()),
        Column('roundness', Float()),
        Column('sphericity', Float()),
        Column('organic', Boolean()),
        Column('scolor', String()),
        Column('smarker', String())
    )

    # Create the table in the database.
    metadata.create_all(engine)
