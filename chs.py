import ast
import pickle
import csv
from datetime import datetime
from itertools import islice
from geotechnipy import cptc
from geotechnipy import chamber
from geotechnipy import soil
from math import ceil
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy import and_, between
import warnings


class CHS():
    """Instantiate the CC CPT, Chamber, and Soil class."""

    def __init__(
        self,
        cid=None,
        hid=None,
        sid=None,
        bcondition=None,
        cdiameter=None,
        cresistance=None,
        chpressure=None,
        cppressure=None,
        dratio=None,
        hthstress=None,
        htvstress=None,
        hppressure=None,
        hevstress=None,
        hehstress=None,
        htmstress=None,
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
        saturation=None,
        sparameter=None,
        sresistance=None,
        vratio=None,
        hdiameter=None,
        hheight=None,
        intercept=None,
        compressibility=None,
        gradation=None,
        diameter10=None,
        diameter30=None,
        diameter50=None,
        diameter60=None,
        curvature=None,
        uniformity=None,
        fines=None,
        organic=None,
        llimit=None,
        plimit=None,
        pindex=None,
        shape=None,
        sphericity=None,
        roundness=None,
        sgravity=None,
        voidx=None,
        voidn=None

    ):
        """
        Instantiate the CC CPT (C), Chamber (H), and Soil class (S).

        The user should note that if the parameter is a list it should be
        specified in the form [min, max].

        Parameters
        ----------
        cid:  list  | calibration chamber CPT ID
        hid:  list  | calibration chamber ID
        sid:  list  | soil ID
        boundary:  list | boundary conditions (see table below)
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
        cdiameter:  list | cone diameter (cm)
        chpressure:  bool | True if CPT lateral stress (kPa) versus depth
            attached; False otherwise
        compressibility:  float | slope of the Critical State Line; although
            the geotechnical convention is for compressibility to be positive,
            it should be input as a negative number here
        cppressure:  bool | True if CPT pore pressure (kPa) versus depth
            attached; False otherwise
        cresistance:  bool | True if cone resistance (MPa) versus depth
            attached; False otherwise
        diameter10:  list | particle diameter at 10% passing
        diameter30:  list | particle diameter at 30% passing
        diameter50:  list | particle diameter at 50% passing
        diameter60:  list | particle diameter at 60% passing
        fines:  list | fines content in decimal
        gradation:  Bool | indicates if sieve analysis results are attached
        dratio:  list | chamber diameter:cone diameter
        hdiameter:  list | chamber diameter (cm)
        hheight:  list | chamber height (cm)
        hid:  list | calibration chamber ID
        hthstress:  list | chamber total horizontal stress (kPa)
        htvstress:  list | chamber total vertical stress (kPa)
        hppressure:  list | calibration chamber pore pressure
        intercept:  list | intercept of the Critical State Line
        llimit:  list | liquid limit (%)
        oratio:  list | overconsolidation ratio
        organic:  Bool | True if the soil is organic (e.g., peat), False
            otherwise
        plimit:  list | plastic limit (%)
        pindex:  list | plasticity index (%)
        rdensity:  list | relative density (decimal)
        roundness:  list | particle roundness
        rvcresistance:  float | representative value of cone resistance (MPa)
            for the test
        rvcresistance_n:  float | representative value of net cone
            resistance (MPa)
        rvdeltaq:  list | representative value of DeltaQ for the test
        rvsresistance:  list | representative value of sleeve resistance (kPa)
        saturation:  bool | True if the soil is saturated; False otherwise
        sgravity:  list | specific gravity
        shape:  list | particle shape (see table below)
            --------------------------------------
            | 1       | rounded                  |
            | 2       | subrounded               |
            | 3       | subangular               |
            | 4       | angular                  |
            | 5       | rounded to subrounded    |
            | 6       | rounded to subangular    |
            | 7       | rounded to angular       |
            | 8       | subrounded to subangular |
            | 9       | subrounded to angular    |
            | 10      | subangular to angular    |
            | -999999 | Not assigned             |
            --------------------------------------
        sparameter:  list | state parameter (decimal)
        sphericity:  list | particle sphericity
        sresistance:  bool | True if sleeve resistance (kPa) versus depth
            attached; False otherwise
        uscs:  int | Unified Soils Classification System classification; can
            be...
        voidx:  list | maximum void ratio
        voidn:  list | minimum void ratio
        vratio:  list | void ratio (decimal)

        """
        engine = create_engine('sqlite:///db/geotechnipy.sqlite')

        connection = engine.connect()

        metadata = MetaData()

        ctable = Table(
            'cptc',
            metadata,
            autoload=True,
            autoload_with=engine
        )

        htable = Table(
            'chamber',
            metadata,
            autoload=True,
            autoload_with=engine
        )

        stable = Table(
            'soil',
            metadata,
            autoload=True,
            autoload_with=engine
        )

        jtable = ctable.join(
            htable,
            ctable.columns.hid==htable.columns.hid
        )
        jtable = jtable.join(
            stable,
            ctable.columns.sid==stable.columns.sid
        )

        # Filter the Soils Database.
        stmt = select([jtable])

        # Filter by ID.
        if cid:
            stmt = stmt.where(ctable.columns.cid.in_((cid)))

        if hid:
            stmt = stmt.where(htable.columns.hid.in_((hid)))

        if sid:
            stmt = stmt.where(stable.columns.sid.in_((sid)))

        # Filter by boundary condition.
        if bcondition:
            stmt = stmt.where(ctable.columns.bcondition.in_(bcondition))

        # Filter by cone diameter.
        if cdiameter:
            stmt = stmt.where(
                between(
                    ctable.columns.cdiameter,
                    cdiameter[0],
                    cdiameter[1]
                )
            )

        # Filter by whether or not cone resistance readings are attached.
        if cresistance:
            stmt = stmt.where(ctable.columns.cresistance.in_(cresistance))

        # Filter by whether or not sleeve resistance reading are attached.
        if sresistance:
            stmt = stmt.where(ctable.columns.sresistance.in_(sresistance))

        # Filter by whether or not cone horitzontal pressure readings are
        # attached.
        if chpressure:
            stmt = stmt.where(ctable.columns.chpressure.in_(chpressure))

        # Filter by whether or not cone pore pressure readings are attached.
        if cppressure:
            stmt = stmt.where(ctable.columns.cppressure.in_(cppressure))

        # Filter by diameter ratio.
        if dratio:
            stmt = stmt.where(
                between(
                    ctable.columns.dratio,
                    dratio[0],
                    dratio[1]
                )
            )

        # Filter by chamber total horizontal stress.
        if hthstress:
            stmt = stmt.where(
                between(
                    ctable.columns.hthstress,
                    hthstress[0],
                    hthstress[1]
                )
            )

        # Filter by chamber total vertical stress.
        if htvstress:
            stmt = stmt.where(
                between(
                    ctable.columns.htvstress,
                    htvstress[0],
                    htvstress[1]
                )
            )

        # Filter by chamber mean total stress.
        if htmstress:
            stmt = stmt.where(
                between(
                    ctable.columns.htmstress,
                    htmstress[0],
                    htmstress[1]
                )
            )

        # Filter by chamber mean effective stress.
        if hemstress:
            stmt = stmt.where(
                between(
                    ctable.columns.hemstress,
                    hemstress[0],
                    hemstress[1]
                )
            )

        # Filter by chamber pore pressure.
        if hppressure:
            stmt = stmt.where(
                between(
                    ctable.columns.hppressure,
                    hppressure[0],
                    hppressure[1]
                )
            )

        # Filter by overconsolidation ratio.
        if oratio:
            stmt = stmt.where(
                between(
                    ctable.columns.oratio,
                    oratio[0],
                    oratio[1]
                )
            )

        # Filter by cone resistance.
        if rvcresistance:
            stmt = stmt.where(
                between(
                    ctable.columns.rvcresistance,
                    rvcresistance[0],
                    rvcresistance[1]
                )
            )

        # Filter by mean effective-stress normalized cone resistance.
        if rvcresistance_mesnn:
            stmt = stmt.where(
                between(
                    ctable.columns.rvcresistance_mesnn,
                    rvcresistance_mesnn[0],
                    rvcresistance_mesnn[1]
                )
            )

        # Filter by net cone resistance.
        if rvcresistance_n:
            stmt = stmt.where(
                between(
                    ctable.columns.rvcresistance_n,
                    rvcresistance_n[0],
                    rvcresistance_n[1]
                )
            )

        # Filter by overburden stress-normalized net cone resistance.
        if rvcresistance_osnn:
            stmt = stmt.where(
                between(
                    ctable.columns.rvcresistance_osnn,
                    rvcresistance_osnn[0],
                    rvcresistance_osnn[1]
                )
            )

        # Filter by relative density.
        if rdensity:
            stmt = stmt.where(
                between(
                    ctable.columns.rdensity,
                    rdensity[0],
                    rdensity[1]
                )
            )

        # Filter by sleeve resistance.
        if rvsresistance:
            stmt = stmt.where(
                between(
                    ctable.columns.rvsresistance,
                    rvsresistance[0],
                    rvsresistance[1]
                )
            )

        # Filter by Delta Q.
        if rvdeltaq:
            stmt = stmt.where(
                between(
                    ctable.columns.rvdeltaq,
                    rvdeltaq[0],
                    rvdeltaq[1]
                )
            )

        # Filter by friction ratio.
        if rvfratio:
            stmt = stmt.where(
                between(
                    ctable.columns.rvfratio,
                    rvfratio[0],
                    rvfratio[1]
                )
            )

        # Filter by whether or not the test was saturated.
        if saturation:
            stmt = stmt.where(ctable.columns.saturation.in_(saturation))

        # Filter by state parameter.
        if sparameter:
            stmt = stmt.where(
                between(
                    ctable.columns.sparameter,
                    sparameter[0],
                    sparameter[1]
                )
            )

        # Filter by void ratio.
        if vratio:
            stmt = stmt.where(
                between(
                    ctable.columns.vratio,
                    vratio[0],
                    vratio[1]
                )
            )

        # Filter by chamber diameter.
        if hdiameter:
            stmt = stmt.where(
                between(
                    htable.columns.hdiameter,
                    hdiameter[0],
                    hdiameter[1]
                )
            )

        # Filter by chamber height.
        if hheight:
            stmt = stmt.where(
                between(
                    htable.columns.hheight,
                    hheight[0],
                    hheight[1]
                )
            )

        # Filter by compressibility.
        if compressibility:
            stmt = stmt.where(
                between(
                    stable.columns.compressibility,
                    compressibility[0],
                    compressibility[1]
                )
            )

        # Filter by CSL intercept.
        if intercept:
            stmt = stmt.where(
                between(
                    stable.columns.intercept,
                    intercept[0],
                    intercept[1]
                )
            )

        # Filter by maximum void ratio.
        if voidx:
            stmt = stmt.where(
                between(
                    stable.columns.voidx,
                    voidx[0],
                    voidx[1]
                )
            )

        # Filter by minimum void ratio.
        if voidn:
            stmt = stmt.where(
                between(
                    stable.columns.voidn,
                    voidn[0],
                    voidn[1]
                )
            )

        ## Filter by liquid limit.
        if llimit:
            stmt = stmt.where(
                between(
                    stable.columns.llimit,
                    llimit[0],
                    llimit[1]
                )
            )

        # Filter by plastic limit.
        if plimit:
            stmt = stmt.where(
                between(
                    stable.columns.plimit,
                    plimit[0],
                    plimit[1]
                )
            )

        # Filter by plasticity index.
        if pindex:
            stmt = stmt.where(
                between(
                    stable.columns.pindex,
                    pindex[0],
                    pindex[1]
                )
            )

        # Filter by specific gravity.
        if sgravity:
            stmt = stmt.where(
                between(
                    stable.columns.sgravity,
                    sgravity[0],
                    sgravity[1]
                )
            )

        # Filter by USCS. <--Coming soon

        # Filter by whether there is a gradation associated with the soil.
        if gradation:
            stmt = stmt.where(stable.columns.gradation.in_(gradation))

        # Filter by particle diameter size.
        if diameter10:
            stmt = stmt.where(
                between(
                    stable.columns.diameter10,
                    diameter10[0],
                    diameter10[1]
                )
            )

        # Filter diameter at 30% passing.
        if diameter30:
            stmt = stmt.where(
                between(
                    stable.columns.diameter30,
                    diameter30[0],
                    diameter30[1]
                )
            )

        # Filter diameter at 50% passing.
        if diameter50:
            stmt = stmt.where(
                between(
                    stable.columns.diameter50,
                    diameter50[0],
                    diameter50[1]
                )
            )

        # Filter diameter at 60% passing.
        if diameter60:
            stmt = stmt.where(
                between(
                    stable.columns.diameter60,
                    diameter60[0],
                    diameter60[1]
                )
            )

        # Filter by Coefficient of Curvature.
        if curvature:
            stmt = stmt.where(
                between(
                    stable.columns.curvature,
                    curvature[0],
                    curvature[1]
                )
            )

        # Filter by Coefficient of Uniformity.
        if uniformity:
            stmt = stmt.where(
                between(
                    stable.columns.uniformity,
                    uniformity[0],
                    uniformity[1]
                )
            )

        # Filter by percent fines.
        if fines:
            stmt = stmt.where(
                between(
                    stable.columns.fines,
                    fines[0],
                    fines[1]
                )
            )

        # Filter by roundness.
        if roundness:
            stmt = stmt.where(
                between(
                    stable.columns.roundness,
                    roundness[0],
                    roundness[1]
                )
            )

        # Filter by particle shape.
        if shape:
            stmt = stmt.where(stable.columns.shape.in_(shape))

        # Filter by sphericity.
        if sphericity:
            stmt = stmt.where(
                between(
                    stable.columns.sphericity,
                    sphericity[0],
                    sphericity[1]
                )
            )

        # Filter by whether the soil is organic or not.
        if organic:
            stmt = stmt.where(stable.columns.organic.in_(organic))

        # Make the dataframe.
        df = pd.read_sql(
            stmt,
            connection
        ).set_index('cid')

        # Note:  joining the tables causes duplicate columns.
        # Remove duplicate columns.
        df = df.T.groupby(level=0).first().T

        # Update the representative value of Delta Q in CHS table with the
        # mean value calculated from calibration chamber CPT tests.
        # Remove entries that do not have a representative value of Delta Q
        # obtained from a calibration chamber CPT test and get a unique
        # list of soil IDs from the results.
        sids = df.dropna(subset=['rvdeltaq']).loc[:, 'sid'].unique().tolist()

        # Iterate through the soils to determine the average Delta Q and
        # assign that value to each soil.
        for sid in sids:

            # Get the Soil table.
            stable = soil.Soil(sid=[sid])

            # Check if the soil has a Delta Q explicitly assigned.
            if stable.results.at[sid, 'deltaq']:
                warnings.warn('deltaq has been explicitly defined for {}. '
                    'The explicitly defined value will be used rather than '
                    'the average value from calibration chamber CPT '
                    'data.'.format(sid)
                )

            else:
                # Get the average value of Delta Q for the soil.
                deltaq = df.loc[df['sid'] == sid, 'rvdeltaq'].mean()

                # Update Delta Q in the Soil table.
                soil.update_soil(sid, deltaq=deltaq)

                # Update the current CHS table.
                df.loc[df['sid'] == sid, 'deltaq'] = deltaq

        self.__results = df


    @property
    def results(self):
        return self.__results

    @property
    def snames(self):
        return self.__results['sname'].unique().tolist()

    @property
    def sids(self):
        return self.__results['sid'].unique().tolist()

    @property
    def rvcresistance(self):
        return self.__results['rvcresistance']

    @property
    def rvcresistance_n(self):
        return self.__results['rvcresistance_n']

    @property
    def rvcresistance_osnn(self):
        return self.__results['rvcresistance_osnn']

    @property
    def rvcresistance_mesnn(self):
        return self.__results['rvcresistance_mesnn']

    @property
    def hevstress(self):
        return self.__results['hevstress']

    def forecast_sparameter_deltaq(self):
        """
        Forecast state parameter from Delta Q.

        This forecasts state parameter using a Delta Q-based correlation.  The
        correlation is defined as

        sparameter_dq =  -0.15 * log(rvdeltaq) + c

        where:
        sparameter_deltaq = state parameter
        rvdeltaq = representative value of Delta Q
        c = -0.191 * log(rvcresistance_n) + 0.381
        rvcresistance_n = representative value of net cone resistance (MPa)

        Parameters
        ----------
        None

        Returns
        -------
        sparameter_deltaq:  Pandas series | state parameter forecasted using
            Delta Q

        """
        c = (
            -0.191 *
            self.__results['rvcresistance_n'].astype('float64')
            .apply(np.log10) + 0.381
        )

        sparameter_deltaq = (
            -0.15 *
            self.__results['rvdeltaq'].astype('float64').apply(np.log10) + c
        )

        return sparameter_deltaq

    def calc_osccoefs(
            self,
            using='sparameter'
    ):
        """
        Calculate the overburden stress correction coefficients.

        The overburden stress correction coefficients can be calculated using
        either state parameter or relative density. If using state parameter,
        the equations are

        Parameters
        ----------
        using:  str | can be 'sprameter' or 'rdensity'

        Returns
        -------
        c:  float | overburden coefficient
        n:  float | overburden coefficient

        """
        # Make empty dataframe
        osccoefs = pd.DataFrame()

        # Calculate the slope and intercept for the equation of stress
        # coefficients C and n
        if using == 'sparameter':
            osccoefs['mc'] = (
                59.38953137979327 *
                np.exp(
                    -3.277963883385083 *
                    self.__results['sparameter'].astype('float')
                )
            )

            osccoefs['bc'] = (
                -191.19047619047626 *
                self.__results['sparameter'].astype('float') +
                27.578571428571394
            )

            osccoefs['mn'] = (
                -1.0749158558295722 *
                np.exp(
                    7.839352177779257 *
                    self.__results['sparameter'].astype('float')
                )
            )

            osccoefs['bn'] = (
                0.5904761904761906 *
                self.__results['sparameter'] +
                0.7010000000000002
            )

        elif using == 'rdensity':
            osccoefs['mc'] = (
                30.190661668283102 *
                np.exp(
                    1.2794916813083883 *
                    self.__results['rdensity'].astype('float')
                )
            )

            osccoefs['bc'] = (
                -0.27849999999999997 *
                self.__results['rdensity'].astype('float') +
                0.7861
            )

            osccoefs['mn'] = (
                -2.594541171943934 *
                np.exp(
                    -3.160251818017144 *
                    self.__results['rdensity'].astype('float')
                )
            )

            osccoefs['bn'] = (
                -0.27849999999999997 *
                self.__results['rdensity'] +
                0.7861
            )

        # Calculate stress coefficients C and n
        osccoefs['c'] = (
            osccoefs['mc'] *
            self.results['compressibility'] +
            osccoefs['bc']
        )

        osccoefs['n'] = (
            osccoefs['mn'] *
            self.__results['compressibility'].astype('float') +
            osccoefs['bn']
        )

        # n must 1 or less; replace n's that are greater than 1
        osccoefs.loc[osccoefs['n'] > 1] = 1

        return osccoefs

    def calc_oscorrection(
        self,
        using='sparameter',
        apressure=100.0
    ):
        """
        Calculate the overburden correction factor.

        Parameters
        ----------
        using:  str | can be 'sprameter' or 'rdensity'
        apressure:  float | atmospheric pressure; generally
          estimated as 100 kPa

        Returns
        -------
        oscorrection:  DataFrame | data frame containing the overburden stress
            correction factor and the normalized representative value of cone
            resistance

        """
        # Get the overburden coefficients
        osccoefs = self.calc_osccoefs()

        # Join the dataframes
        df = pd.concat([self.__results, osccoefs], axis=1)

        # Make empty data frame
        oscorrection = pd.DataFrame()

        # Calculate the overburden stress correction factor (i.e., Cn)
        df['oscfactor'] = (100 / df['hevstress'])**df['n']

        # The correction factor is limited to 1.7 or less
        df.loc[df['oscfactor'] > 1.7] = 1.7

        # Calculate the normalized representative value of cone resistance
        df['nrvcresistance'] = df['rvcresistance'] * df['oscfactor']

        # Get the results
        oscorrection = df[['oscfactor', 'nrvcresistance']]

        return oscorrection

    def calc_rvic_rw98(self):
        """
        Calculate soil behavior type index.

        For more information on soil behavior type index, Ic, see Robertson
        and Wride (1998), 'Evaluating liquefaction potential using the cone
        penetration test'.

        Parameters
        ----------
        None

        Returns
        -------
        rvic_rw98:  Pandas series| representative value of soil behavior type
            index

        """
        rvic_rw98 = np.sqrt(
            (
                3.47 -
                self.__results['rvcresistance_osnn'].astype('float64')\
                .apply(np.log10)
            )**2 +
            (
                self.__results['rvfratio'].astype('float64').apply(np.log10) +
                1.22
            )**2
        )

        return rvic_rw98

    def calc_rvn_sbt(self):
        """
        Calculate the stress coefficient.

        This function calculates the stress coefficient (which is sometimes
        referred to at the stress exponent) using the Soil Behavior Type
        concept.  See Robertson (2009), 'Interpretation of cone penetration
        tests--a unified approach' for more information.

        The stress coefficient is limited to values less than or equal to 1.

        Parameters
        ----------
        None

        Returns
        -------
        rvn_sbt:  Pandas series | representative value of stress coefficient
            (decimal)

        """
        rvic_rw98 = self.calc_rvic_rw98()

        rvn_sbt = (
            0.381 * rvic_rw98 +
            0.05 * (self.__results['hevstress'] / 101.325) -
            0.15
        )

        rvn_sbt[rvn_sbt > 1] = 1.

        return rvn_sbt

    def calc_rvcparameter(self):
        """
        Calculate the normalized cone parameter.

        This function calculates the normalized cone parameter.  See Robertson
        (2009), 'Interpretation of cone penetration tests--a unified approach'
        for more information.

        Parameters
        ----------
        None

        Returns
        -------
        rvcparameter:  float | representative value of cone parameter (no
            units)

        """
        rvn_sbt = self.calc_rvn_sbt()

        rvcparameter = (
            (self.__results['rvcresistance_n'] * 1000 / 101.325) *
            (101.325 / self.__results['hevstress'])**rvn_sbt
        )

        return rvcparameter

    def visualize(
        self,
        xdata,
        ydata,
        c='k',
        s=8,
        **kwargs
    ):
        """
        Visually explore data in the CHS database.

        xdata:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        ydata:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        c:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        s:  str | can be any of the columns in the soil, chamber, and cptc
            tables

        Returns
        -------
        None

        """
        # Check if the specified data is int64 or float64.
        #if self.__results[xdata] not (
        #    np.int64 or
        #    np.float64
        #):
        #    warnings.warn("""The specified data is not numeric and cannot be
        #    plotted.
        #    """)
        #x =


    def visualize_cs_space(
        self,
        c='scolor',
        s=8,
        save=False,
        fname=None,
        **kwargs
    ):
        """
        Plot Critical State space.

        c:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        s:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        save:  bool | True to save; False otherwise
        fname:  str | file name

        Returns
        -------
        None

        """
        sids = self.__results['sid'].unique()

        for sid in sids:

            print(sid)

            df = self.__results[self.__results['sid'] == sid]
            c = df['scolor'].tolist()[0]
            marker = df['smarker'].tolist()[0]
            compressibility = df['compressibility'].tolist()[0]
            intercept = df['intercept'].tolist()[0]

            fig1, ax1 = make_axes_cstate()

            ax1.plot(
                [0, 0],
                [0.2, 1.4]
            )

            # CPTc data
            ax1.scatter(
                df['hehstress'],
                df['vratio'],
                alpha=0.65,
                c=c,
                edgecolors='black',
                marker=marker,
                zorder=2
            )

            xs = np.linspace(1, 1000)
            ys = compressibility * np.log10(np.linspace(1, 1000)) + intercept

            # Critical State Line
            ax1.plot(
                xs,
                ys,
                color='k',
                zorder=1
            )

            # Data range for equal state parameter lines
            ymax = max(ys[0], df['vratio'].max())
            ymin = min(ys[-1], df['vratio'].min())
            xmax = max(1000, df['hehstress'].max())
            xmin = min(1, df['hehstress'].min())

            sparameterx = df['sparameter'].max()
            sparametern = df['sparameter'].min()

            # Plot equal state lines above the CSL
            if (
                sparameterx > 0 and
                ceil(sparameterx / 0.06) > 0
            ):

                for count in range(1, ceil(sparameterx / 0.06) + 1):

                    # Intercept for equal state line
                    esplintercept = intercept + (0.06 * count)

                    # Data range
                    ymax_ = min(esplintercept, ymax)
                    xmin_ = (ymax_ + esplintercept) / compressibility

                    xs = np.linspace(xmin, xmax)
                    ys = compressibility * np.log10(xs) + esplintercept

                    ax1.plot(
                        xs,
                        ys,
                        color='0.65',
                        linestyle='--',
                        zorder=1
                    )

            # Plot equal state lines below the CSL
            if (
                sparametern < 0 and
                ceil(abs(sparametern / 0.06)) > 0
            ):

                for count in range(1, ceil(abs(sparametern / 0.06)) + 1):

                    # Intercept for equal state line
                    esplintercept = intercept - (0.06 * count)

                    # Data range
                    xs = np.linspace(xmin, xmax)
                    ys = compressibility * np.log10(xs) + esplintercept

                    ax1.plot(
                        xs,
                        ys,
                        color='0.65',
                        linestyle='--',
                        zorder=1
                    )

            plt.show()

    def visualize_cresistance(
        self,
        using='rvcresistance',
        ftype='jr1c',
        save=False,
        fname=None,
        legend=False
    ):
        """
        Plot cone ressistance versus effective vertical stress.

        Parameters
        ----------
        using:  str | can be any of the following variants of cone resistance
            -------------------------------------------
            | rvcresistance       | cone resistance                 |
            | rvcresistance_n     | net cone resistance             |
            | rvcresistance_osnn  | overburden stress-normalized    |
            |                     |    cone resistance              |
            | rvcresistance_mesnn | mean effective stress-normalize |
            |                     |    cone resistance
        ftype:  str | figure type, can be one of the following
            ---------------------------------------
            | Type     | Width (in) | Height (in) |
            ---------------------------------------
            | 'dr'     | 6          | 4           |
            | 'ds'     | 6          | 6           |
            | 'jr1c'   | 3.5        | 2.33        |
            | 'jr1.5c' | 5.5        | 3.66        |
            | 'jr2c'   | 7.5        | 5           |
            | 'js1c'   | 3.5        | 3.5         |
            ---------------------------------------
            d = dissertation
            j = journal
            r = rectangular
            s = square
            c = column
        save:  Bool | True to save; False otherwise
        fname:  str | file name
        legend:  bool | True to show legend, false otherwise

        Returns
        -------
        fig1:  Matplotlib figure object
        ax1:  Matplotlib axis object

        """
        # Figure and axis objects
        fig1, ax1 = plt.subplots(figsize=figsize[ftype])

        # Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=size[ftype])

        # Axis
        xlabel = {
            'rvcresistance': 'Cone resistance, $q_\\mathrm{t}$, MPa',
            'rvcresistance_n': (
                'Net cone resistance, $q_\\mathrm{t} - '
                '\\sigma_\\mathrm{v0}$, MPa'
            ),
            'rvcresistance_osnn': (
                'Overburden stress-normalized net cone resistance, '
                '$\\frac{q_\\mathrm{t} - \\sigma^{\\prime}_\\mathrm{v0}}'
                '{\\sigma^{\\prime}_mathrm{v0}}$'
            ),
            'rvcresistance_mesnn': (
                'Mean effective-stress normalized net cone resistance, '
                '$\\frac{q_\\mathrm{t} - p}{p^{\\prime}}$'
            )
        }
        ax1.set_xlabel(xlabel[using])
        ax1.set_ylabel(
            'Effective vertical stress, $\\sigma^{\\prime}_\\mathrm{v0}$, kPa'
        )
        ax1.invert_yaxis()

        # Tick marks
        ax1.minorticks_on()
        ax1.tick_params(which='both', top=True, right=True)

        # Get location of soil color (cloc), soil name (lloc), and soil marker
        # (mloc)
        cloc = self.__results.columns.get_loc('scolor')
        lloc = self.__results.columns.get_loc('sname')
        mloc = self.__results.columns.get_loc('smarker')

        for sid in self.sids:

            df = self.__results[self.__results['sid'] == sid]

            ax1.scatter(
                df[using],
                df['hevstress'],
                alpha=0.65,
                c=df.iat[0, cloc],
                edgecolors='black',
                label=df.iat[0, lloc],
                marker=df.iat[0, mloc]
            )

        if legend:
            ax1.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.2),
                ncol=2
            )

        if save:

            if not fname:
                time = datetime.now()
                fname = 'visualize_cresistance_{:%Y%m%d}_{:%H%M%S}'.\
                    format(time, time)

            time = datetime.now()
            plt.savefig(
                'figs/{}.png'.format(fname),
                bbox_inches='tight',
                dpi=300
            )
            plt.savefig(
                'figs/{}.pdf'.format(fname),
                bbox_inches='tight',
                dpi=300
            )

        plt.show()

        return fig1, ax1


    def visualize_deltaq(
        self,
        c='scolor',
        s=8,
        save=False,
        fname=None,
        **kwargs
    ):
        """
        Plot Delta Q space.

        Parameters
        ----------
        c:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        s:  str | can be any of the columns in the soil, chamber, and cptc
            tables
        save:  bool | True to save; False otherwise
        fname:  str | file name

        Returns
        -------
        None

        """
        # Get the figure and axes.
        fig1, ax1 = axes_deltaq()

        # Normalize the size data so that it's not too big or too small.
        if type(s) == int:
            s = s**2

        else:
            # Get data.
            sdata = self.__results[s]

            # Get maximum and minimum values.
            vmax = sdata.max()
            vmin = sdata.min()

            # Normalize the data to the maximum and minimum values between
            # values of 6 and 12.
            s = sdata.apply(
                lambda x: (((vmax - x) / (vmax - vmin)) * (12 - 6) + 6)**2
            )

        # Prepare the plotting parameters for marker color.
        if c == 'scolor':
            cdata = self.__results['scolor']

            # Set the color map and normalization.
            cmap = None
            norm = None

        # Use a qualitative color map for certain data.
        elif (
            c == 'shape' or
            c == 'uscs'
        ):
            cdata = self.__results[c]

            # Set the color map
            cmap = cm.Set1

            # Set the normalization.
            norm = None

        # Use a sequential color map for the remaining data.
        else:
            # Get the data.
            cdata = self.__results[c]

            # Set the color map.
            cmap = cm.viridis

            # Get maximum and minimum values.
            vmax = cdata.max()
            vmin = cdata.min()

            # Normalize the data.
            norm = cm.colors.Normalize(vmax=vmax, vmin=vmin)

        scatter = ax1.scatter(
            self.__results['rvsresistance_esn'],
            self.__results['rvcresistance_osnn'],
            alpha=0.65,
            c=cdata,
            cmap=cmap,
            s=s,
            edgecolor='k',
            linewidth=0.15,
            norm=norm
        )

        # Make the colorbar
        if (
            c != 'scolor' and
            c == 'shape'
        ):
            cbar = fig1.colorbar(
                scatter,
                ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
            cbar.ax.set_yticklabels(
                [
                    'Rounded', 'Subrounded', 'Subangular', 'Angular',
                    'Rounded to Subrounded', 'Rounded to Subangular',
                    'Rounded to Angular', 'Subrounded to Subangular',
                    'Subrounded to Angular', 'Subangular to Angular'
                ]
                )
            cbar.set_label('Shape')

        elif c != 'scolor':
            l = label[c]

            plt.colorbar(scatter, label=l)

        if save:

            if not fname:
                time  = datetime.now()
                fname = 'visualize_deltaq_{:%Y%m%d}_{:%H%M%S}'.\
                    format(time, time)

            time  = datetime.now()
            plt.savefig(
                'figs/{}.png'.format(fname),
                bbox_inches='tight',
                dpi=300
            )
            plt.savefig(
                'figs/{}.pdf'.format(fname),
                bbox_inches='tight',
                dpi=300
            )

        return fig1, ax1

    def visualize_dratio(
        self,
        versus='rvcresistance_mesnn',
        legend=False,
        save=False,
        fname=False,
        ftype='jr1c'
    ):
        """
        Plot diameter ratio versus mean-stress-normalized cone resistance.

        Parameters
        ----------
        versus:  str | can be any
        legend:  bool | True to show legend, false otherwise
        save:  bool | True to save, false otherwise
        fname:  str | file name
        ftype:  str | figure type, can be one of the following
            ---------------------------------------
            | Type     | Width (in) | Height (in) |
            ---------------------------------------
            | 'dr'     | 6          | 4           |
            | 'ds'     | 6          | 6           |
            | 'jr1c'   | 3.5        | 2.33        |
            | 'jr1.5c' | 5.5        | 3.66        |
            | 'jr2c'   | 7.5        | 5           |
            | 'js1c'   | 3.5        | 3.5         |
            ---------------------------------------
            d = dissertation
            j = journal
            r = rectangular
            s = square
            c = column

        Returns
        -------
        None

        """
        # Figure and axis objects
        fig1, ax1 = plt.subplots(figsize=figsize[ftype])

        # Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=size[ftype])

        # Axis labels
        ax1.set_xlabel('Diameter ratio, $d_\\mathrm{r}$')
        ax1.set_ylabel(r'$(q_\mathrm{t} - p) / p^\prime$')

        # Tick marks
        ax1.minorticks_on()
        ax1.tick_params(which='both', top=True, right=True)

        # Plotting range
        #ax1.set_xlim([10, 70])

        # Soil IDs
        sids = self.__results['sid'].unique().tolist()

        # Get location of soil color (cloc), soil name (lloc), and soil marker
        # (mloc)
        cloc = self.__results.columns.get_loc('scolor')
        lloc = self.__results.columns.get_loc('sname')
        mloc = self.__results.columns.get_loc('smarker')

        for sid in sids:

            df = self.__results[self.__results['sid'] == sid]

            ax1.scatter(
                df['dratio'],
                df[versus],
                alpha=0.25,
                c=df.iat[0, cloc],
                edgecolors='black',
                label=df.iat[0, lloc],
                marker=df.iat[0, mloc],
                zorder=1
            )

            # Box and whisker plot
            for dratio in df['dratio'].unique().tolist():

                # Select y data at each unique value of dratio
                df2 = df[df['dratio']==dratio]

                if len(df2[versus]) <= 1:
                    continue

                # Make width equal to 20 pixels.
                # Note:  Here, I'm converting two points ([0, 0], [20, 0]),
                # which are in pixel coordinates and converting them to axes
                # coordinates.  Then I'm finding the distance between those
                # points which is then the width of the box plot.
                points = ax1.transData.inverted().transform([(0, 0), (20, 0)])
                point1 = points[0, 0]
                point2 = points[1, 0]
                width = point2 - point1

                ax1.boxplot(
                    df2[versus],
                    manage_ticks=False,
                    positions=[dratio],
                    widths=width,
                    showfliers=False
                )

        if legend:
            ax1.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.2),
                ncol=2
            )

        if save:

            if not fname:
                time = datetime.now()
                fname = 'dratio_v_rvcresistance_mesnn_{:%Y%m%d}_{:%H%M%S}'.\
                    format(time, time)

            plt.savefig(
                'figs/{}.png'.format(fname),
                bbox_inches='tight',
                dpi=600
            )

            plt.savefig(
                'figs/{}.pdf'.format(fname),
                bbox_inches='tight',
                dpi=600
            )

        plt.show()

        return fig1, ax1

    def visualize_sparameter_v_rvcresistance_mesnn(
        fname=False,
        ftype='jr1c',
        save=False,
    ):
        """
        Plot state parameter versus mean stress-normalized cone resistance.

        Parameters
        ----------
        fname:  str | file name
        ftype:  str | figure type, can be one of the following
            ---------------------------------------
            | Type     | Width (in) | Height (in) |
            ---------------------------------------
            | 'dr'     | 6          | 4           |
            | 'ds'     | 6          | 6           |
            | 'jr1c'   | 3.5        | 2.33        |
            | 'jr1.5c' | 5.5        | 3.66        |
            | 'jr2c'   | 7.5        | 5           |
            | 'js1c'   | 3.5        | 3.5         |
            ---------------------------------------
            d = dissertation
            j = journal
            r = rectangular
            s = square
            c = column
        save:  bool | True to save, false otherwise

        Returns
        -------
        fig1:  Matplotlib figure object
        ax1:  Matplotlib axis object

        """
        # Figure and axis objects
        fig1, ax1 = plt.subplots(figsize=figsize[ftype])

        # Font
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=size[ftype])

        # Face color
        ax1.set_facecolor('0.95')

        # Axis labels
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(r'$(q_\mathrm{c} - p) / p^\prime$')

        # Tick marks
        ax1.minorticks_on()
        ax1.tick_params(which='both', top=True, right=True)

        # Location of soil information
        cloc = self.__results.columns.get_loc('scolor')
        lloc = self.__results.columns.get_loc('sname')
        mloc = self.__results.columns.get_loc('smarker')

        ax1.scatter(
            df['dratio'],
            df['rvcresistance_mesnn'],
            alpha=0.65,
            c=df.iat[0, cloc],
            edgecolors='black',
            label=df.iat[0, lloc],
            marker=df.iat[0, mloc]
        )

        if save:

            if not fname:
                time = datetime.now()
                fname = 'sparameter_v_rvcresistance_mesnn_{:%Y%m%d}_{:%H%M%S}'.\
                    format(time, time)

            plt.savefig(
                'figs/{}.png'.format(fname),
                bbox_inches='tight',
                dpi=600
            )

            plt.savefig(
                'figs/{}.pdf'.format(fname),
                bbox_inches='tight'
            )

        plt.show()


class CPTcOld():
    """Work with cone penetration test data from the calibration chamber."""

    def __init__(
            self,
            cptc_loc,
            sdb_loc
    ):
        """
        Work with cone penetration test data from the calibration chamber.

        Parameters
        ----------
        cptc_loc:  str | path to the CPT Calibrabation Chamber Database
        sdb_loc:  str | path to the Soil Database

        """
        with open(cptc_loc, 'rb') as in_file:
            cptc = pickle.load(in_file)

        self.__dict__ = cptc

        # Initialize the location of the CPT Calibration Chamber Database as a
        # class variable.
        self.__cptc_loc = cptc_loc

        # Initialize the location of the Soil Database as a class variable.
        self.__sdb_loc = sdb_loc

    @property
    def ids(self):
        """Get ID numbers of CPTs in the Calibration Chamber Database."""
        # Get list of CPT IDs.
        tests = list(self.__dict__.keys())

        # Remove extraneous entries in the object dictionary.
        # Note:  When the CPT Database class is instantiated, the path to the
        # Soil Database is stored in the class dictionary.  Since we want the
        # test IDs only--which are also the  keys to the dictionary--then we
        # must remove the path.
        tests.remove('_CPTc__cptc_loc')
        tests.remove('_CPTc__sdb_loc')

        return tests

    def __str__(self):
        """Print the number of tests loaded in the CPT database."""
        string = '''{} tests loaded.'''

        return string.format(len(self.__dict__))

    def compressibility_correction(
        self,
        cpt_id
    ):
        """
        Calculate the compressibility correction factor.

        Parameters
        ----------
        cpt_id:  Str | CPT ID number

        Returns
        -------
        ccf: float | compressibility correction factor

        """
        # Instantiate the Soil Database.
        sdb = soil.SDb(self.__sdb_loc)

        # Get the relative density of the calibration chamber test.
        relative_density = self.__dict__[cpt_id]['relative_density']

        # Get the soil ID of the calibration chamber test.
        soil_id = self.__dict__[cpt_id]['soil_id']

        # Get the compressibility of the calibration chamber test.
        compressibility = sdb.__dict__[soil_id]['csl_slope']

        # Change the compressibility from negative to positive.
        # Note:  the geotechnical convention is for compressibility to be
        # positive.
        compressibility = compressibility * -1

        # Get a representative value of tip resistance for the calibration
        # chamber test.
        tip_resistance = self.representative_value(
            cpt_id,
            self.__dict__[cpt_id]['tip_resistances']
        )

        # Get the vertical effective stress of the calibration chamber test.
        vertical_effective_stress = self.chamber_vertical_effective_stress(
            cpt_id
        )

        # Initialize variables to find the least compressible soil.
        compressibility_lcs = -999999
        soil_id_lcs = '__none__'

        # Find the least compressible soil.
        for soil_id_ in sdb.ids:
            if sdb.__dict__[soil_id_]['csl_slope'] > compressibility_lcs:
                compressibility_lcs = sdb.__dict__[soil_id_]['csl_slope']
                soil_id_lcs = soil_id_

        # Change the compressibility from negative to positive.
        # Note:  the geotechnical convention is for compressibility to be
        # positive.
        compressibility_lcs = compressibility_lcs * -1

        # Calculate the slope, m, and intercept, b, for the relative density of
        # the calibration chamber test.
        mc = - 0.851 * relative_density - 31.290
        bc = 0.702 * relative_density + 8.580

        # Calculate stress coefficient C for the calibration chamber test.
        c = mc * compressibility + bc

        # Calculate stress coefficient C for the least compressible soil
        c_lcs = mc * compressibility_lcs + bc

        # Calculate the slope, m, and intercept, b, for the relative density of
        # the calibration chamber test.
        mn = - 0.851 * np.log(relative_density) - 3.989
        bn = -0.002 * relative_density + 0.763

        # Calculate stress coefficient n for the calibration chamber test.
        n = mn * compressibility + bn

        # Calculate stress coefficient n for the least compressible soil.
        n_lcs = mn * compressibility_lcs + bn

        # Forecast the tip resistance for the calibration chamber test.
        tip_resistance = c * vertical_effective_stress**n

        # Forecast the tip resistance for the least compressible soil.
        tip_resistance_lcs = c_lcs * vertical_effective_stress**n_lcs

        # Calculate the compressibility correction factor.
        ccf = tip_resistance_lcs / tip_resistance

        return (ccf, soil_id_lcs)

    def diam_correction_factor_go18(
        self,
        cpt_id
    ):
        """
        Calculate the diameter correction for calibration chamber tests.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        correction_factor:  float | Correction factor based on diameter ratio
            and CSL slope

        """
        # Get state parameter.
        state_parameter = self.__dict__[cpt_id]['state_parameter']

        # Make state parameter bins.
        state_param_bins = [-0.36, -0.30, -0.24, -0.18, -0.12, -0.06, 0]

        # Get state parameter bin number.
        bin_num = np.digitize(state_parameter, state_param_bins).tolist()

        # Make state parameter bin dictionary.
        state_param_dict = {
            1: '[-0.36, -0.30)',
            2: '[-0.30, -0.24)',
            3: '[-0.24, -0.18)',
            4: '[-0.18, -0.12)',
            5: '[-0.12, -0.06)',
            6: '[-0.06, 0)'
        }

        # Check if the state parameter is within the ranges given above (i.e,
        # -0.36 <= state_parameter < 0)
        # NOTE: If the bin number is less than 1, then the state parameter of
        # the test is outside the ranges given above.  These ranges have not
        # been analyzed for correction factors.  Thus, 1 is returned.
        if (
            bin_num < 1 or
            bin_num > 6
        ):
            return 1, 1

        # Get state parameter label.
        state_parameter_bin_label = state_param_dict[bin_num]

        # Get soil id.
        soil_id = self.__dict__[cpt_id]['soil_id']

        # Instantiate the Soil Database.
        sdb = soil.SDb(self.__sdb_loc)

        # Get the CSL Slope.
        csl_slope = sdb.__dict__[soil_id]['csl_slope']

        # Make CSL slope bins.
        csl_slope_bins = [-0.20, -0.16, -0.12, -0.08, -0.04, -0]

        # Get CSL slope bin number.
        bin_num = np.digitize(csl_slope, csl_slope_bins).tolist()

        # Make CSL slope dictionary.
        csl_slope_dict = {
            1: '[-0.20, -0.16)',
            2: '[-0.16, -0.12)',
            3: '[-0.12, -0.08)',
            4: '[-0.08, -0.04)',
            5: '[-0.04, 0)'
        }

        # Get CSL bin label.
        csl_slope_bin_label = csl_slope_dict[bin_num]

        # Get boundary_condition.
        bc = self.__dict__[cpt_id]['boundary_condition']

        # Get diameter correction coefficients dictionary.
        if (
            bc == 1 or
            bc == 4
        ):
            name = 'bc14_fit_parameters.dat'
        elif (
            bc == 2 or
            bc == 3
        ):
            name = 'bc23_fit_parameters.dat'
        else:
            return 1, 1

        with open(name) as in_file:
            csv.reader(in_file, delimiter=',')
            coefs = ast.literal_eval(in_file.read())

        # Get diameter correction coefficients.
        l1 = coefs[csl_slope_bin_label][state_parameter_bin_label]['l1']
        l2 = coefs[csl_slope_bin_label][state_parameter_bin_label]['l2']
        k = coefs[csl_slope_bin_label][state_parameter_bin_label]['k']
        x0 = coefs[csl_slope_bin_label][state_parameter_bin_label]['x0']

        # Get measured mean-stress-normalized tip resistance.
        measured_msn_tip_resistance = (
            self.msnn_tip_resistance(cpt_id)
        )

        # Calculate the diameter correction factor.
        if measured_msn_tip_resistance > l1:
            correction_factor = 1

        elif measured_msn_tip_resistance < l2:
            correction_factor = l1/l2

        elif l2 <= measured_msn_tip_resistance <= l1:

            # Get the diameter ratio.
            diameter_ratio = self.diameter_ratio(cpt_id)

            # Calculate the mean stress normalized tip resistance.
            estimated_msn_tip_resistance = (
                (l1 - l2) / (1 + np.exp(k * (diameter_ratio - x0))) + l2
            )

            correction_factor = l1 / estimated_msn_tip_resistance

        # NOTE:  Been et al. (1986) give two trend lines to estimate the
        # diameter correction factor:  one corresponding to a range of state
        # parameter greater than -0.05 and the other corresponding to a range
        # of state parameter between -0.29 and -0.27.

        # For instances where the state parameter is in between the given
        # ranges, the code below interpolates trend lines for state parameters
        # that are not in the given ranges (i.e., greater than -0.05 or between
        # -0.29 and -0.27).  For instances where the state parameter is less
        # than the given ranges (i.e., less than -0.29) the code below
        # extrapolates the trend line by determining the rate at which the
        # mean-stress-normalized tip resistance changes with respect to state
        # parameter.

        # The numbers/variables below (e.g., xi, xf) were obtained as follows:
        # 198.374 and 596.166   mean-stress-normalized tip resistance at the
        #                       left- and right-most point on the inclined
        #                       portion of the trend line for BCs 1 and 4 with
        #                       state parameter in the range of -0.29 and -0.27
        # 468.211 and 596.166   mean-stress-normalized tip resistance at the
        #                       left- and right-most point on the inclined
        #                       portion of the trend line for BCs 2 and 3 with
        #                       state parameter in the range of -0.29 and -0.27
        # 80                    mean-stress-normalized tip resistance for the
        #                       trend line for BCs 1, 2, 3, and 4 with state
        #                       parameter greater -0.05
        # 20.2978 and 50        diameter ratio at the left- and right-most
        #                       point on the inclined portion of the trend
        #                       line for BCs 1 and 4

        # Get yi and yf (the initial and final points on the inclined portion
        # of the trend lines proposed by Been et al. 1986).
        if (
            bc == 1 or
            bc == 4
        ):
            yi = 198.374
            yf = 596.166

        elif (
            bc == 2 or
            bc == 3
        ):
            yi = 468.211
            yf = 596.166

        # Get the mean-stress-normalized tip resistance as estimated by the
        # trend lines from Been et al. (1986).
        if state_parameter >= -0.05:
            y1 = 80
            y2 = 80

        elif -0.29 < state_parameter < -0.05:
            y1 = yi
            y2 = yf

        elif -0.29 <= state_parameter <= -0.27:
            y1 = yi
            y2 = yf

        elif state_parameter <= -0.29:

            # Calculate the rate at which the mean-stress-normalized tip
            # changes with respect to state parameter at the left-most point
            # of the trend lines.
            rate1 = (yi - 80) / (-0.29 - -0.05)

            # Calculate the rate at which the mean-stress-normalized tip
            # changes with respect to state parameter at the right-most point
            # of the trend lines.
            rate2 = (yf - 80) / (-0.29 - -0.05)

            y1 = state_parameter * rate1
            y2 = state_parameter * rate2

        # Calculate trend line slope.
        slope = (y2 - y1) / (50 - 20.2978)

        # Calculate trend line intercept.
        intercept = y1 - slope * 20.2978

        # Get diameter ratio.
        diameter_ratio = self.diameter_ratio(cpt_id)

        # Calculate estimated mean-stress-normalized tip resistance using
        # the trend lines from Been et al. (1986).
        if diameter_ratio <= 50:
            emsn_tip_resistance = slope * diameter_ratio + intercept

        elif diameter_ratio > 50:
            emsn_tip_resistance = y2

        # Calculate the Been et al. (1986) correction factor.
        correction_factor_bea86 = y2 / emsn_tip_resistance

        return correction_factor, correction_factor_bea86

    def diam_correction_factor_bea86(
        self,
        cpt_id
    ):
        """
        Calculate the diameter correction for calibration chamber tests.

        Been et al. (1986) give two trend lines to estimate the diameter
        correction factor:  one corresponding to a range of state parameter
        greater than -0.05 and the other corresponding to a range of state
        parameter between -0.29 and -0.27.

        For instances where the state parameter is in between the given ranges,
        the code below interpolates trend lines for state parameters that are
        not in the given ranges (i.e., greater than -0.05 or between -0.29 and
        -0.27).  For instances where the state parameter is less than the given
        ranges (i.e., less than -0.29) the code below extrapolates the trend
        line by determining the rate at which the mean-stress-normalized tip
        resistance changes with respect to state parameter.

        The numbers/variables used below (e.g., xi, xf) were obtained as
        follows:

            198.374 and 596.166
            -------------------
            Mean-stress-normalized tip resistance at the left- and right-most
            point on the inclined portion of the trend line for BCs 1 and 4
            with state parameter in the range of -0.29 and -0.27.

            468.211 and 596.166
            -------------------
            mean-stress-normalized tip resistance at the left- and right-most
            point on the inclined portion of the trend line for BCs 2 and 3
            with state parameter in the range of -0.29 and -0.27.

            80
            --
            Mean-stress-normalized tip resistance for the trend line for BCs 1,
            2, 3, and 4 with state parameter greater -0.05

            20.2978 and 50
            --------------
            Diameter ratio at the left- and right-most point on the inclined
            portion of the trend line for BCs 1 and 4.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        correction_factor:  float | Correction factor based on diameter ratio
            and CSL slope

        """
        # Get state parameter.
        state_parameter = self.__dict__[cpt_id]['state_parameter']

        # Get boundary_condition.
        bc = self.__dict__[cpt_id]['boundary_condition']

        # Get yi and yf (the initial and final points on the inclined portion
        # of the trend lines proposed by Been et al. 1986).
        if (
            bc == 1 or
            bc == 4
        ):
            yi = 198.374
            yf = 596.166

        elif (
            bc == 2 or
            bc == 3
        ):
            yi = 468.211
            yf = 596.166

        # Get the mean-stress-normalized tip resistance as estimated by the
        # trend lines from Been et al. (1986).
        if state_parameter >= -0.05:
            y1 = 80
            y2 = 80

        elif -0.29 < state_parameter < -0.05:
            y1 = yi
            y2 = yf

        elif -0.29 <= state_parameter <= -0.27:
            y1 = yi
            y2 = yf

        elif state_parameter <= -0.29:

            # Calculate the rate at which the mean-stress-normalized tip
            # changes with respect to state parameter at the left-most point
            # of the trend lines.
            rate1 = (yi - 80) / (-0.29 - -0.05)

            # Calculate the rate at which the mean-stress-normalized tip
            # changes with respect to state parameter at the right-most point
            # of the trend lines.
            rate2 = (yf - 80) / (-0.29 - -0.05)

            y1 = state_parameter * rate1
            y2 = state_parameter * rate2

        # Calculate trend line slope.
        slope = (y2 - y1) / (50 - 20.2978)

        # Calculate trend line intercept.
        intercept = y1 - slope * 20.2978

        # Get diameter ratio.
        diameter_ratio = self.diameter_ratio(cpt_id)

        # Calculate estimated mean-stress-normalized tip resistance using
        # the trend lines from Been et al. (1986).
        if diameter_ratio <= 50:
            emsn_tip_resistance = slope * diameter_ratio + intercept

        elif diameter_ratio > 50:
            emsn_tip_resistance = y2

        # Calculate the Been et al. (1986) correction factor.
        correction_factor = y2 / emsn_tip_resistance

        return correction_factor

    def diam_correction_factor_jea01(
        self,
        cpt_id
    ):
        """
        Calculate diameter correction factor using Jamiolkowsi et al. 2001.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        correction_factor:  float | diameter correction factor

        """
        # Get diameter ratio..
        diameter_ratio = self.diameter_ratio(cpt_id)

        # Get relative density.
        relative_density = self.__dict__[cpt_id]['relative_density']

        # Get data from table 3 of Jamiolkowsi et al. 2001.
        diameter_ratios = [22.1, 33.6, 47.2, 60, 100]
        a_coefs = [0.054, 0.090, 0.166, 0.412, 1]
        b_coefs = [0.827, 0.624, 0.457, 0.221, 0]
        min_relative_densities = [0.341, 0.471, 0.508, 0.558, 1]

        # Get interpolated value of minimum relative density.
        min_relative_density = np.interp(
            diameter_ratio,
            diameter_ratios,
            min_relative_densities
        )

        # Check relative density.
        # NOTE:  An assumption of this equation is that there are no
        # boundary effects if, given a certain diameter ratio, the relative
        # density of the CPT sounding is less than the minimum relative
        # density.
        if relative_density <= min_relative_density:
            correction_factor = 1
            return correction_factor

        # Get the interpolated value of the a coefficient.
        a_coef = np.interp(
            diameter_ratio,
            diameter_ratios,
            a_coefs
        )

        # Get the interpolated value of the b coefficient.
        b_coef = np.interp(
            diameter_ratio,
            diameter_ratios,
            b_coefs
        )

        # Get the boundary conditions.
        bc = self.__dict__[cpt_id]['boundary_condition']

        # Get the m coefficient.
        if (
            bc == 1 or
            bc == 4
        ):
            m_coef = 1

        elif (
            bc == 2 or
            bc == 3
        ):
            m_coef = -1

        else:
            m_coef = -999999

        # Calculate correction factor
        correction_factor = a_coef * ((relative_density * 100)**b_coef)**m_coef

        return correction_factor

    def diam_correction_factor_mk91(
        self,
        cpt_id,
    ):
        """
        Calculate diameter correction factor using Mayne and Kulhawy (1991).

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        correction_factor:  float | diameter correction factor

        """
        # Get diameter ratio.
        diameter_ratio = self.diameter_ratio(cpt_id)

        # Check diameter ratio.
        # NOTE:  An assumption of this equation is that there are no boundary
        # effects when the value of diameter ratio is greater than or equal to
        # 70.
        if diameter_ratio >= 70:
            correction_factor = 1

            return correction_factor

        # Get relative density.
        relative_density = self.__dict__[cpt_id]['relative_density']

        correction_factor = (
            (diameter_ratio - 1) / 70
            )**-(relative_density / 200)

        return correction_factor

    def diameter_correction_viz(
            self,
            bcs,
            csls_bin_label,
            sp_bin_label,
            save='no',
            xlim=(0, 120)
           ):
        """
        Plot diameter ratio versus mean-stress-normalized cone resistance.

        Parameters
        ----------
        bcs:  list of int | base and side restraint conditions
            for a calibration chamber test; can be any of those given below

            ----------------------------------------------------------
            | Boundary Condition | Side Restraint  | Base Restraint  |
            ----------------------------------------------------------
            | 1                  | Constant Stress | Constant Stress |
            | 2                  | Constant Volume | Constant Volume |
            | 3                  | Constant Volume | Constant Stress |
            | 4                  | Constant Stress | Constant Volume |
            ----------------------------------------------------------

        csls_bin_label:  int | label of the bin containing the CSL
            slope of interest; must be selected from the following:

            ------------------------
            | Bin | CSL Slope      |
            ------------------------
            | 1   | [-0.04, 0)     |
            | 2   | [-0.08, -0.04) |
            | 3   | [-0.12, -0.08) |
            | 4   | [-0.16, -0.12) |
            | 5   | [-0.20, -0.16) |
            ------------------------

        sp_bin_label:  list of str | label of the bin containing the state
            parameter of interest; must be selected from the following:

            -------------------------
            | Bin | State Parameter |
            -------------------------
            | A   | [-0.06, 0)      |
            | B   | [-0.12, -0.06)  |
            | C   | [-0.18, -0.12)  |
            | D   | [-0.24, -0.18)  |
            | E   | [-0.30, -0.24)  |
            | F   | [-0.36, -0.30)  |
            -------------------------

        save_location:  str | 'yes' to save
        xlim: tuple of int | range of trend line

        Returns
        -------
        intercept:  float | intercept of the linear trend line used to model
            the data
        slope:  float | slope of the linear tredn line used to model the data

        """
        # Get calibration chamber database.
        DF = self.get_data_by_bin(
            bcs,
            csls_bin_label,
            sp_bin_label
        )

        # ---Make figure object and set figure attributes---
        # Make figure and axis objects.
        fig1, ax1 = plt.subplots(figsize=(6, 4))

        # Set font.
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)

        # Set face color.
        ax1.set_facecolor('0.95')

        # Set the axis labels.
        ax1.set_xlabel(r'$d_\mathrm{chamber} / d_\mathrm{cone}$')
        ax1.set_ylabel(r'$q_\mathrm{c} - p / p^\prime$')

        # Set the minor ticks.
        ax1.minorticks_on()

        # ---Make scatter plot--- #
        # If there are data points, make the scatter plot.
        if DF.empty is not True:
            ax1.scatter(
                DF['diameter_ratio'],
                DF['msn_tip_resistance'],
                alpha=0.5,
                c='#e41a1c',
                edgecolors='black',
                label='__nolegend__',
                linewidths=0.5,
                marker='o',
                s=8**2,
                zorder=2
            )

        # ---Make box and whisker plot---
        # If there is only one data point, do nothing.
        if (
            DF.empty is not True and
            DF['diameter_ratio'].size > 1
        ):

            # Get unique locations of x data.
            locs = set(DF['diameter_ratio'])

            for loc in locs:

                # Get data at loc.
                df = DF[DF['diameter_ratio'] == loc]

                (
                    q1,
                    q2,
                    q3,
                    min_msn_tip_resistance,
                    max_msn_tip_resistance
                ) = bw_plot(df)

                # Make box 20 points wide (10 points * 2).
                x1, y1 = ax1.transData.inverted().transform((0, 0))
                x2, y2 = ax1.transData.inverted().transform((10, 0))
                half_width = x2 - x1

                # Plot box.
                ax1.add_patch(
                    patches.Rectangle(
                        [loc - half_width, q1],
                        half_width * 2,
                        q3 - q1,
                        alpha=0.75,
                        edgecolor='black',
                        facecolor='#4daf4a',
                        linewidth=0.5,
                        zorder=1
                    )
                )

                # Plot median.
                ax1.plot(
                    [loc - half_width, loc + half_width],
                    [q2, q2],
                    alpha=0.75,
                    color='black',
                    label='__nolegend__',
                    linewidth=0.5,
                    zorder=1
                )

                # Plot lower fence.
                ax1.plot(
                    [loc - half_width, loc + half_width],
                    [min_msn_tip_resistance, min_msn_tip_resistance],
                    alpha=0.75,
                    color='black',
                    label='__nolegend__',
                    linewidth=0.5,
                    zorder=1
                )

                # Plot upper fence.
                ax1.plot(
                    [loc - half_width, loc + half_width],
                    [max_msn_tip_resistance, max_msn_tip_resistance],
                    alpha=0.75,
                    color='black',
                    label='__nolegend__',
                    linewidth=0.5,
                    zorder=1
                )

                # Plot lower whisker.
                ax1.plot(
                    [loc, loc],
                    [q1, min_msn_tip_resistance],
                    alpha=0.75,
                    color='black',
                    label='__nolegend__',
                    linewidth=0.5,
                    zorder=1
                )

                # Plot upper whisker.
                ax1.plot(
                    [loc, loc],
                    [q3, max_msn_tip_resistance],
                    alpha=0.75,
                    color='black',
                    label='__nolegend__',
                    linewidth=0.5,
                    zorder=1
                )

        # ---Plot proposed diameter correction--- #
        # Make x data for the trend line.
        XS = np.linspace(xlim[0], xlim[1], 200)

        # Get the diameter correction trend line coeffifients.
        l1, l2, k, x0 = get_diameter_correction_coefs(
            bcs,
            csls_bin_label,
            sp_bin_label
        )

        # Get y data for the trend line.
        ys = diameter_correction_func(
            XS,
            l1,
            l2,
            k,
            x0
        )

        # Plot proposed diameter correction.
        ax1.plot(
            XS,
            ys,
            color='magenta',
            label=(
                'Diameter correction (This study)'
            )
        )

        # ---Plot Been et al. (1986) diameter correction--- #
        # Get slope and intercept of Been et al. (1986) trend line.
        slope, intercept, y2 = get_bea86_diameter_correction_coefs(
            bcs,
            sp_bin_label
        )

        # Get y data points for Been et al. (1986) trend line.
        ys = bea86_diameter_correction_func(
            XS,
            slope,
            intercept,
            y2
        )

        ax1.plot(
            XS,
            ys,
            color='cyan',
            label=(
                'Diameter correction (Been et al. 1986)'
            )
        )

        # ---Make the legend--- #
        ax1.legend(loc='lower right')

        # Save plot if prompted.
        if save is 'yes':

            # Make the file name.
            fname = 'diameter_correction_{}{}.pdf'.format(
                csls_bin_label,
                sp_bin_label
            )

            plt.savefig(
                fname,
                bbox_inches='tight'
            )

        # Show the plot.
        plt.show()



    def coef_lep(
        self,
        cpt_id
    ):
        """
        Calculate the coefficient of lateral earth pressure.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        coef_lep:  float |coefficient of lateral earth pressure

        """
        coef_lep = (
            self.chamber_lateral_effective_stress(cpt_id) /
            self.chamber_lateral_effective_stress(cpt_id)
        )

        return coef_lep

    def get_data(
        self,
        bcs=[1, 2, 3, 4, 5],
        csls_range=[-999999, 999999],
        soil_ids='all',
        sp_range=[-999999, 999999]
    ):
        """
        Filter calibration chamber data by certain parameters.

        This function is used to filter the calibration chamber database for
        plotting.  Thus, it does not include all the information contained in
        the calibration chamber database.

        Parameters
        ----------
        bcs:  list of int | base and side restraint conditions
            for a calibration chamber test; can be any of those given below

            ----------------------------------------------------------
            | Boundary Condition | Side Restraint  | Base Restraint  |
            ----------------------------------------------------------
            | 1                  | Constant Stress | Constant Stress |
            | 2                  | Constant Volume | Constant Volume |
            | 3                  | Constant Volume | Constant Stress |
            | 4                  | Constant Stress | Constant Volume |
            ----------------------------------------------------------

        csls_range:  list of float | None or list in the form
            (min CSL slope, max CSL slope)

        sp_range:  list of float | None or list in the form
            (min state parameter, max state parameter)

        soil_ids:  list of str | None or soil ID

        Returns
        -------
        df:  DataFrame | standard geotechnipy calibration chamber database

        """
        # Instantiate the Soil Database.
        sdb = soil.SDb(self.__sdb_loc)

        # Initialize empty dictionary for plot data.
        d = {}

        # Get plot data.
        for cpt_id in self.ids:

            # Get boundary condition.
            boundary_condition = self.__dict__[cpt_id]['boundary_condition']

            # Filter by boundary condition.
            if boundary_condition in bcs:
                pass

            else:
                continue

            # Get soil ID.
            soil_id = self.__dict__[cpt_id]['soil_id']

            # Filter by soil ID.
            if (
                soil_ids is 'all' or
                soil_id in soil_ids
            ):
                pass

            else:
                continue

            # Get slope of the critical state line.
            csl_slope = sdb.__dict__[soil_id]['csl_slope']

            # Get CSL intercept.
            csl_intercept = sdb.__dict__[soil_id]['csl_intercept']

            csls_min, csls_max = csls_range

            # Filter by CSL slope.
            if csls_min <= csl_slope <= csls_max:
                pass

            else:
                continue

            # Get state parameter.
            state_parameter = self.__dict__[cpt_id]['state_parameter']

            sp_min, sp_max = sp_range

            # Filter by state parameter
            if sp_min <= state_parameter <= sp_max:
                pass

            else:
                continue

            # Get chamber diameter ratio.
            diameter_ratio = self.diameter_ratio(cpt_id)

            # Get mean-effective-stress-normalized cone resistance.
            mesnn_tip_resistance = (
                    self.mesnn_tip_resistance(cpt_id)
            )

            # Get mean total and effective stress.
            mean_total_stress = self.mean_total_stress(cpt_id)
            mean_effective_stress = self.mean_effective_stress(cpt_id)

            # Get tip resistance.
            tip_resistance = self.representative_value(
                cpt_id,
                self.__dict__[cpt_id]['tip_resistances']
            )

            # Convert mean total stress to MPa and calculate the mean net
            # penetration resistance.
            mn_tip_resistance = tip_resistance - mean_total_stress / 1000

            # Get the mean sleeve friction.
            sleeve_friction = self.representative_value(
                cpt_id,
                self.__dict__[cpt_id]['sleeve_frictions']
            )

            # Get delta q.
            if sleeve_friction == -999999:
                delta_q = -999999
            else:
                delta_q = self.delta_q(cpt_id)

            # Get chamber lateral effective stress.
            chamber_lateral_effective_stress = (
                self.chamber_lateral_effective_stress(cpt_id)
            )

            # Get voids ratio.
            voids_ratio = sdb.sp_2_vr(
                chamber_lateral_effective_stress,
                soil_id,
                state_parameter
            )

            # Make dictionary for plotting.
            d[cpt_id] = {
                'boundary_condition': boundary_condition,
                'cpt_name': self.__dict__[cpt_id]['cpt_name'],
                'csl_slope':  csl_slope,
                'csl_intercept': csl_intercept,
                'cpt_color': self.__dict__[cpt_id]['color'],
                'delta_q': delta_q,
                'diameter_ratio': diameter_ratio,
                'marker': self.__dict__[cpt_id]['marker'],
                'mean_effective_stress': mean_effective_stress,
                'mesnn_tip_resistance': mesnn_tip_resistance,
                'mn_tip_resistance': mn_tip_resistance,
                'reference_name': self.__dict__[cpt_id]['reference_name'],
                'reference_year': self.__dict__[cpt_id]['reference_year'],
                'sleeve_friction': sleeve_friction,
                'soil_color': sdb.__dict__[soil_id]['soil_color'],
                'soil_id':  soil_id,
                'soil_marker': sdb.__dict__[soil_id]['marker_type'],
                'soil_name': sdb.__dict__[soil_id]['soil_name'],
                'state_parameter': state_parameter,
                'voids_ratio': voids_ratio
            }

        # Make dataframe from dictionary.
        df = pd.DataFrame.from_dict(d, orient='index')

        return df


    def msnoc_tip_resistance(
        self,
        cpt_id
    ):
        """
        Calculate effective mean stress normalized overburden tip resistance.

        The effective-mean-stress normalized tip resistance, [q_c1]_P', is
        defined by the equation

        [q_c1]_P' = (q_c1 - P) / P'

        where

        q_c1    = overburden corrected tip Resistance
        P       = (sigma_1 + sigma_3 * 2) / 3
        P'      = (sigma_1' + sigma_3' * 2) / 3

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        mean_stress_normalized_tip_resistance:  list of float | normalized,
            corrected tip resistance

        """
        # Get tip resistance and convert from MPa to kPa.
        tip_resistance = np.array(
            self.__dict__[cpt_id]['tip_resistances']
        ) * 1000

        # Get the overburden corrrection factor.
        oc_factor = self.oc_factor(cpt_id)

        # Calculate the overburden corrected tip resistance.
        oc_tip_resistance = tip_resistance * oc_factor

        # Get total stress.
        mean_total_stress = np.array(self.mean_total_stress(cpt_id))

        # Get effective stress.
        mean_effective_stress = np.array(self.mean_effective_stress(cpt_id))

        # Calculate the normalized, corrected tip resistance.
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_stress_normalized_tip_resistances = np.divide(
                (oc_tip_resistance - mean_total_stress),
                mean_effective_stress
                )

            # Replace infinite values, if any, with 0.
            mean_stress_normalized_tip_resistances[
                mean_stress_normalized_tip_resistances == np.inf
            ] = 0

            # Replace nans, if any, with 0.
            mean_stress_normalized_tip_resistances = np.nan_to_num(
                mean_stress_normalized_tip_resistances
                )

        # Calculate the representative value of the mean stress
        # normalized tip resistance.
        mean_stress_normalized_tip_resistance = self.representative_value(
            cpt_id,
            mean_stress_normalized_tip_resistances
        )

        return mean_stress_normalized_tip_resistance


def bw_plot(df):
    """
    Get x and y coordinates to draw a box and whisker plot.

    This helper function is intended only to be used with calibration chamber
    functions.

    Parameters
    ----------
    df:  DataFrame | standard geotechnipy calibration chamber dataframe

    Returns
    -------
    q1:  float | y value of the first quartile
    q2:  float | y value of the second quartile
    q3:  float | y value of the third quartile
    min_msn_tip_resistance:  float | minimum mean-stress-normalized tip
        resistance
    max_msn_tip_resistance:  float | maximum mean-stress-normalized tip
        resistance

    """
    # Get lower quartile (q1), median (q2), and upper quartile (q3) of the
    # mean-stress-normalized tip resistance.
    q1 = df['msn_tip_resistance'].quantile(
        q=0.25,
        interpolation='lower',
    )

    q2 = df['msn_tip_resistance'].quantile(
        q=0.5,
        interpolation='midpoint',
    )

    q3 = df['msn_tip_resistance'].quantile(
        q=0.75,
        interpolation='higher',
    )

    # Get min and max mean stress normalized tip resistance.
    min_msn_tip_resistance = df['msn_tip_resistance'].min()
    max_msn_tip_resistance = df['msn_tip_resistance'].max()

    return q1, q2, q3, min_msn_tip_resistance, max_msn_tip_resistance


def get_cdb(cptccdb_loc):
    """
    Get the CPT Database.

    Parameters
    ----------
    cptccdb_loc:  str | path to the CPT Calibration Chamber Database

    Returns
    -------
    cptccdb:  dict | CPT Calibration Chamber Database
    """
    with open(cptccdb_loc, 'rb') as in_file:
        cptccdb = pickle.load(in_file)

    return cptccdb


def cpt_file(
    file_name,
    marker_color,
    marker_type,
    *points,
    cone_diameter=3.57,
    soil_group='not given',
    soil_name='not given',
    water_type='fresh'
):
    """
    Loads a .CPT file from the Golder Associates online repository into the
    CPT Database.

    Parameters
    ----------
    cone_diameter:  float | cone diameter in cm
    file_name:  str | file name
    marker_color:  str | marker color; should be one of the following:
                         '#a6cee3'
                         '#1f78b4'
                         '#b2df8a'
                         '#33a02c'
                         '#fb9a99'
                         '#e31a1c'
                         '#fdbf6f'
                         '#ff7f00'
                         '#cab2d6'
                         '#6a3d9a'
                         '#ffff99'
                         '#b15928'
    marker_type:  str | marker type; should be one of the following:
                        'o'
                        's'
                        'D'
                        'p'
                        '*'
                        'P'
                        'X'
                        '^'
                        'v'
                        '<'
                        '>'
    rel_dense:  float | relative density
    soil_group:  str | soil group (e.g., 'Ottawa')
    soil_name:  str | soil name (e.g., 'Ottawa 40/70')
    points:  tup | tuple in the form (depth to top of soil layer, unit weight)
    water_type: str | can be 'fresh' (9.81 kN/m^3) or 'sea' (10.02 kN/m^3)

    Returns
    -------
    None
    """
    # Get header.
    with open(
        'C:/Users/Joseph/Documents/CPTs/{}/{}'
        .format(soil_group, file_name), 'r'
    ) as in_file:
        reader = csv.reader(in_file, delimiter='\t')
        header = [i for i in islice(reader, 0, 10)]

    # Get header data.
    location = header[0][0][10:].strip()
    cpt_name = header[1][0][10:].strip().lower()
    test_date = header[2][0][13:].strip()
    cone_type = header[3][0][10:].strip()
    msl = -float(header[4][0][16:].strip())
    cols = header[9][0][18:].lstrip().rstrip().rstrip('.').split('  ')

    cpt_id = ''.join(char for char in cpt_name.lower() if char.isalnum())

    # Parse columns.
    cols_dict = {
        'Depth (m)': 'depth',
        'Qc Tip Resistance (MPa)': 'tip_resistance',
        'PWP (MPa)': 'pore_pressure',
        'Skin Fr (MPa)': 'sleeve_friction',
        'Lateral Stress (MPa)': 'lateral_stress'
    }

    names = []

    for col in cols:
        if col in cols_dict.keys():
            names.append(cols_dict[col])

        else:
            print(col + 'not recognized')

    usecols = []

    for count in range(len(names)):
        usecols.append(count)

    # Get data.
    df = pd.read_csv(
        'C:/Users/Joseph/Documents/CPTs/{}/{}'
        .format(soil_group, file_name),
        delim_whitespace=True,
        index_col=0,
        skiprows=10,
        names=names,
        skip_blank_lines=True,
        usecols=usecols
    )

    # Drop data points where there are duplicate depth readings.
    df = df[~df.index.duplicated(keep='first')]

    # Change the data (except for tip resistance) to kPa.
    for name in names:
        if name != 'tip_resistance':
            df[name] = df[name] * 1000

    # Add 'outlier' data to columns that are absent.
    if 'pore_pressure' not in names:
        df['pore_pressure'] = -999999

    if 'sleeve_friction' not in names:
        df['sleeve_friction'] = -999999

    if 'lateral_stress' not in names:
        df['lateral_stress'] = -999999

    # Add unit weight column to df.
    df['unit_weight'] = 0

    layers = soil_layers(*points)

    for key, subdict in sorted(layers.items()):

        top, bottom = subdict['depth_range']
        unit_weight = subdict['unit_weight']

        # Insert row at the depth given by top.
        df.loc[top] = 0

        # Sort the dataframe.
        df.sort_index(inplace=True)

        # Insert unit weighs according to the soil profile.
        df.loc[top:, ('unit_weight')] = unit_weight

        # Remove the row at the depth given by top.
        df.drop(
            top,
            inplace=True
        )

    # Add total vertical column.
    depths = df.index
    total_stresses = []

    for depth in depths:
        total_stresses.append(total_stress(depth, msl, layers, water_type))

    df['total_stress'] = total_stresses

    # Add pore pressure column.
    df['hydrostatic_pore_pressure'] = map(pore_pressure, depths)

    df['effective_stress'] = (
        df['total_stress'] - df['hydrostatic_pore_pressure']
    )

    # Get CPT Database.
    loc = 'C:/Users/Joseph/Documents/Dissertation/CPT Database/cptdb.dat'

    with open(loc, 'rb') as in_file:
        cptdb = pickle.load(in_file)

    cptdb[cpt_id] = {
        'boundary_condition': -999999,
        'chamber_diameter': -999999,
        'cpt_name': cpt_name,
        'cpt_type': 'field',
        'depth': df.index.tolist(),
        'effective_stress': df['effective_stress'].tolist(),
        'hydrostatic_pore_pressure': df['hydrostatic_pore_pressure'].tolist(),
        'lateral_stress': df['lateral_stress'].tolist(),
        'marker_color': marker_color,
        'marker_type': marker_type,
        'pore_pressure': df['pore_pressure'].tolist(),
        'rel_dens': [-999999],
        'sleeve_friction': df['sleeve_friction'].tolist(),
        'soil_group': soil_group,
        'soil_name': soil_name,
        'state_parameter': -999999,
        'tip_resistance': df['tip_resistance'].tolist(),
        'total_stress': df['total_stress'].tolist(),
        'unit_weight': df['unit_weight'].tolist()
    }

    # Save Soils Database.
    with open(loc, 'wb') as out_file:
        pickle.dump(cptdb, out_file)


def cct(
        cptccdb_loc,
        reference_name,
        reference_year,
        cpt_name,
        saturation,
        tip_resistances,
        chamber_vertical_total_stress,
        soil_id,
        relative_density,
        color,
        marker,
        boundary_condition=-999999,
        chamber_diameter=-999999,
        chamber_height=-999999,
        chamber_lateral_total_stress=-999999,
        chamber_pore_pressure=-999999,
        cone_diameter=3.57,
        cpt_lateral_total_stresses=[[-999999], [-999999]],
        cpt_pore_pressures=[[-999999], [-999999]],
        overconsolidation_ratio=-999999,
        sleeve_frictions=[[-999999], [-999999]],
        state_parameter=-999999
        ):
    """
    Load a calibration chamber test into the CPT database.

    Parameters
    ----------
    cptccbd_loc:  str | path to the CPT Calibration Chamber Database
    reference_name:  str | name of the reference or project
    reference_year:  int | four digit year of the reference or project
    cpt_name:  str | name of the CPT sounding
    soil_id:  str | ID of the soil
    color:  str | marker color given in Hex code
    marker:  str | marker type; should be one of the following:

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

    relative_density:  float | relative density in decimal
    saturation:  bool | True if saturated, False if not saturated
    tip_resistances:  list of lists | tip resistance readings from the CPT in
        the format [[tip resistance in MPa], [depth in m]]
    boundary_condition: int | boundary condition per the table below

          ----------------------------------------------------------
          | Boundary Condition | Side Restraint  | Base Restraint  |
          ----------------------------------------------------------
          | 1                  | Constant Stress | Constant Stress |
          | 2                  | Constant Volume | Constant Volume |
          | 3                  | Constant Volume | Constant Stress |
          | 4                  | Constant Stress | Constant Volume |
          ----------------------------------------------------------

    chamber_diameter:  float | calibration chamber diameter in cm
    chamber_height:  float | calibration chamber height in cm
    chamber_lateral_total_stress:  float | lateral total stress in the
        calibration chamber in kPa
    chamber_pore_pressure:  float | pore pressure in the calibration chamber in
        kPa
    chamber_vertical_total_stress:  float | vertical total stress in the
        calibration chamber in kPa
    cone_diameter:  float | cone penetrometer diameter in cm
    cpt_lateral_total_stresses:  list of lists | total lateral stress readings
        from the CPT in the format [[lateral stress in kPa], [depth in m]]
    cpt_pore_pressures:  list of lists | pore pressure readings from the CPT in
        the format [[pore pressure in kPa], [depth in m]]
    overconsolidation_ratio: float | overconsolidation ratio
    sleeve_frictions:  list of lists | sleeve friction readings from the CPT in
        the format [[sleeve friction in kPa], [depth in m]]

    Returns
    -------
    None

    """
    # Make CPT ID.
    cpt_id = (
        soil_id[:3] +
        '-' +
        ''.join(char for char in str(cpt_name).lower() if char.isalnum()) +
        '-' +
        reference_name[:3].lower() +
        '-' +
        str(reference_year)[2:]
    )

    # Unpack the tip resistance tuple.
    tip_resistances_, depths = tip_resistances

    # Calculate the tip resistance corrected for cone diameter.
    # NOTE
    # The quantity 3.57 refers to the standard cone size, 3.57 cm.
    # tip_resistances_ = [
    #    tip_resistance * 3.57 / cone_diameter
    #    for tip_resistance in tip_resistances_
    # ]

    # Get the length of 'depth'.
    # NOTE
    # The length of 'depth' will be used to format all other lists in this
    # function to make all the data the same length.  That is to say, it will
    # be used to make the rectangular
    length = len(depths)

    # Check if the specimen is saturated.
    if saturation is True:

        # Make a list of hydrostatic pore pressures that is the same length as
        # as the list of depths.
        hydrostatic_pore_pressures = [chamber_pore_pressure] * length

    # Check if the calibration chamber test is unsaturated.
    elif saturation is False:

        # Make a list of hydrostatic pore pressures (i.e., 0) that is the same
        # length as 'depths'.
        hydrostatic_pore_pressures = [0] * length

    # Check if default value was given for CPT lateral stress.
    if cpt_lateral_total_stresses[0] == -999999:

        # Make dummy list of CPT lateral stresses that is the same length as
        # 'depths'.
        cpt_lateral_total_stresses_ = [-999999] * length

    # If CPT lateral stress data was given...
    else:

        # Get CPT lateral stress data at the same depths as 'depth'.
        # NOTE
        # To calculate data from digitized CPT data, the readings must be at
        # the same depth (e.g., if you want to calculate friction ratio at a
        # depths of 1 m, 2 m, 3 m, etc. then the tip resistance and skin
        # friction data must all be at 1 m, 2 m, 3 m, etc.).  This is
        # complicated, though, because digitizing software does not output data
        # in this way.  Thus, the readings will be obtained by interpolating
        # the x data at the depths from the tip resistance data (i.e.,
        # 'depths').
        cpt_lateral_total_stresses_ = np.interp(
            depths,
            cpt_lateral_total_stresses[1],
            cpt_lateral_total_stresses[0]
        )

    # Check if default value was given for lateral stress.
    if cpt_pore_pressures[0] == -999999:

        # Make list of lateral stresses that is the same length as 'depth'.
        cpt_pore_pressures_ = [-999999] * length

    # If CPT pore pressure data was given...
    else:

        # Get CPT pore pressure data at the same depths as 'depth'.
        # NOTE
        # To calculate data from digitized CPT data, the readings must be at
        # the same depth (e.g., if you want to calculate friction ratio at a
        # depths of 1 m, 2 m, 3 m, etc. then the tip resistance and skin
        # friction data must all be at 1 m, 2 m, 3 m, etc.).  This is
        # complicated, though, because digitizing software does not output data
        # in this way.  Thus, the readings will be obtained by interpolating
        # the x data at the depths from the tip resistance data (i.e.,
        # 'depths').
        cpt_pore_pressures_ = np.interp(
                depths,
                cpt_pore_pressures[1],
                cpt_pore_pressures[0]
                )

    # Check if default value was given for sleeve friction.
    if sleeve_frictions[0][0] == -999999:

        # Make a dummy list of sleeve friction that is the same length as
        # 'depth'.
        sleeve_frictions_ = [-999999] * length

    # If sleeve friction data is given...
    else:

        # Get sleeve friction data at the same depths as 'depth'.
        # NOTE
        # To calculapte data from digitized CPT data, the readings must be at
        # the same depth (e.g., if you want to calculate friction ratio at a
        # depths of 1 m, 2 m, 3 m, etc. then the tip resistance and skin
        # friction data must all be at 1 m, 2 m, 3 m, etc.).  This is
        # complicated, though, because digitizing software does not output data
        # in this way.  Thus, the readings will be obtained by interpolating
        # the x data at the depths from the tip resistance data (i.e.,
        # 'depths').
        sleeve_frictions_ = np.interp(
                depths,
                sleeve_frictions[1],
                sleeve_frictions[0])

    # Get the CPT Database.
    with open(cptccdb_loc, 'rb') as in_file:
        cptcc_database = pickle.load(in_file)

    # Update the CPT Database.
    cptcc_database[cpt_id] = {
        'boundary_condition': boundary_condition,
        'chamber_diameter': chamber_diameter,
        'chamber_height':  chamber_height,
        'chamber_lateral_total_stress': chamber_lateral_total_stress,
        'chamber_vertical_total_stress': chamber_vertical_total_stress,
        'cone_diameter': cone_diameter,
        'cpt_name': cpt_name,
        'cpt_pore_pressures': cpt_pore_pressures_,
        'cpt_type': 'chamber',
        'depths': depths,
        'hydrostatic_pore_pressures': hydrostatic_pore_pressures,
        'cpt_lateral_total_stresses': cpt_lateral_total_stresses_,
        'color': color,
        'marker': marker,
        'overconsolidation_ratio': overconsolidation_ratio,
        'relative_density': relative_density,
        'reference_name': reference_name,
        'reference_year': reference_year,
        'sleeve_frictions': sleeve_frictions_,
        'soil_id': soil_id,
        'state_parameter': state_parameter,
        'tip_resistances': tip_resistances_,
    }

    # Save CPT Database.
    with open(cptccdb_loc, 'wb') as out_file:
        pickle.dump(cptcc_database, out_file)


def get_diameter_correction_coefs(
    bcs,
    csls_bin_label,
    sp_bin_label,
):
    """
    Get the diameter correction coefficients.

    Parameters
    ----------
    bcs:  str | boundary condition for the calibration chamber test; must be
        one of the following:  '1, 4' or '2, 3'
    csls_bin_label:  int | label of the bin containing the CSL slope of
        interest; must be selected from the following:

        ------------------------
        | Bin | CSL Slope      |
        ------------------------
        | 1   | [-0.04, 0)     |
        | 2   | [-0.08, -0.04) |
        | 3   | [-0.12, -0.08) |
        | 4   | [-0.16, -0.12) |
        | 5   | [-0.20, -0.16) |
        ------------------------

    sp_bin_label:  str | label of the bin containing the state parameter of
        interest; must be selected from the following:

        -------------------------
        | Bin | State Parameter |
        -------------------------
        | A   | [-0.06, 0)      |
        | B   | [-0.12, -0.06)  |
        | C   | [-0.18, -0.12)  |
        | D   | [-0.24, -0.18)  |
        | E   | [-0.30, -0.24)  |
        | F   | [-0.36, -0.30)  |
        -------------------------

    Returns
    -------
    l1:  float | upper limit
    l2:  float | lower limit
    k:  float | slope
    x0:  horizontal offset

    """
    # Make state parameter bin dictionary.
    sp_bin_dict = {
        'A': '[-0.06, 0)',
        'B': '[-0.12, -0.06)',
        'C': '[-0.18, -0.12)',
        'D': '[-0.24, -0.18)',
        'E': '[-0.30, -0.24)',
        'F': '[-0.36, -0.30)'
    }

    sp_bin_range = sp_bin_dict[sp_bin_label]

    # Make CSL slope bin dictionary.
    csls_bin_dict = {
        1: '[-0.20, -0.16)',
        2: '[-0.16, -0.12)',
        3: '[-0.12, -0.08)',
        4: '[-0.08, -0.04)',
        5: '[-0.04, 0)'
    }

    csls_bin_range = csls_bin_dict[csls_bin_label]

    # Get diameter correction coefficients dictionary.
    if bcs == '1, 4':
        name = 'bc14_fit_parameters.dat'
    elif '2, 3':
        name = 'bc23_fit_parameters.dat'

    with open(name) as in_file:
        csv.reader(in_file, delimiter=',')
        coefs = ast.literal_eval(in_file.read())

    l1 = coefs[csls_bin_range][sp_bin_range]['l1']
    l2 = coefs[csls_bin_range][sp_bin_range]['l2']
    k = coefs[csls_bin_range][sp_bin_range]['k']
    x0 = coefs[csls_bin_range][sp_bin_range]['x0']

    return (l1, l2, k, x0)


def diameter_correction_func(
    xs,
    l1,
    l2,
    k,
    x0
):
    """
    Calculate y values of the proposed diameter correction trend lines.

    Parameters
    ----------
    xs:  list of float | x data for plot
    l1:  float | upper limit
    l2:  float | lower limit
    k:  float | slope
    x0:  horizontal offset

    Returns
    -------
    ys:  list of float | y data for plot

    """
    # Make y data for the trend line.
    ys = [(l1 - l2) / (1 + np.exp(k * (x - x0))) + l2 for x in xs]

    return ys


def get_bea86_diameter_correction_coefs(
    bcs,
    sp_bin_label
):
    """
    Plot the diameter correction from Been et al. (1986).

    Parameters
    ----------
    bcs:  str | boundary condition for the calibration chamber test; must be
        one of the following:  '1, 4' or '2, 3'
    sp_bin_label:  str | label of the bin containing the state parameter of
        interest; must be selected from the following:

        -------------------------
        | Bin | State Parameter |
        -------------------------
        | A   | [-0.06, 0)      |
        | B   | [-0.12, -0.06)  |
        | C   | [-0.18, -0.12)  |
        | D   | [-0.24, -0.18)  |
        | E   | [-0.30, -0.24)  |
        | F   | [-0.36, -0.30)  |
        -------------------------

    Returns
    -------
    slope:  float | slope of the diameter correction trend line
    intercept:  float | intercept of the diameter correction trend line
    xf:  float | value at which the diameter correction levels off

    """
    # Plot Been et al. (1986) diameter correction trend line.
    # Get the data points for the trend line from Been et al. (1986).

    # Note:  Been et al. (1986) give two trend lines to estimate the
    # diameter correction factor:  one corresponding to a range of state
    # parameter greater than -0.05 and the other corresponding to a range
    # of state parameter between -0.29 and -0.27.

    # For instances where the state parameter is in between the given
    # ranges, the code below interpolates trend lines for state parameters
    # that are not in the given ranges (i.e., greater than -0.05 or between
    # -0.29 and -0.27).  For instances where the state parameter is less
    # than the given ranges (i.e., less than -0.29) the code below
    # extrapolates the trend line by determining the rate at which the
    # mean-stress-normalized tip resistance changes with respect to state
    # parameter.

    # The numbers/variables below (e.g., xi, xf) were obtained as follows:
    # 198.374 and 596.166   mean-stress-normalized tip resistance at the
    #                       left- and right-most point on the inclined
    #                       portion of the trend line for BCs 1 and 4 with
    #                       state parameter in the range of -0.29 and -0.27
    # 468.211 and 596.166   mean-stress-normalized tip resistance at the
    #                       left- and right-most point on the inclined
    #                       portion of the trend line for BCs 2 and 3 with
    #                       state parameter in the range of -0.29 and -0.27
    # 80                    mean-stress-normalized tip resistance for the
    #                       trend line for BCs 1, 2, 3, and 4 with state
    #                       parameter greater -0.05
    # 20.2978 and 50        diameter ratio at the left- and right-most
    #                       point on the inclined portion of the trend
    #                       line for BCs 1 and 4

    # Get yi and yf (the initial and final points on the inclined portion
    # of the trend lines proposed by Been et al. 1986).
    if bcs == '1, 4':
        yi = 198.374
        yf = 596.166

    elif bcs == '2, 3':
        yi = 468.211
        yf = 606.504

    # Make state parameter bin dictionary.
    sp_bin_dict = {
        'A': (-0.06, 0),
        'B': (-0.12, -0.06),
        'C': (-0.18, -0.12),
        'D': (-0.24, -0.18),
        'E': (-0.30, -0.24),
        'F': (-0.36, -0.30)
    }

    # Get average state parameter of the bin.
    sp_avg = np.mean(sp_bin_dict[sp_bin_label])

    # Calculate mean-stress-normalized tip resistance, y1 and y2, at
    # diameter ratio = 20.2978 and diameter ratio = 50, respectively.
    if sp_avg >= -0.05:
        y1 = 80
        y2 = 80

    elif -0.27 < sp_avg < -0.05:
        y1 = np.interp(sp_avg, [-0.27, -0.05], [yi, 80])
        y2 = np.interp(sp_avg, [-0.27, -0.05], [yf, 80])

    elif -0.29 <= sp_avg <= -0.27:
        y1 = yi
        y2 = yf

    elif sp_avg < -0.29:
        # Calculate rate at which the mean-stress-normalized tip resistance
        # changes WRT state parameter.
        rate1 = (yi - 80) / (-0.29 - -0.05)
        rate2 = (yf - 80) / (-0.29 - -0.05)

        y1 = sp_avg * rate1
        y2 = sp_avg * rate2

    # Calculate trend line slope.
    slope = (y2 - y1) / (50 - 20.2978)

    # Calculate trend line intercept.
    intercept = y1 - slope * 20.2978

    return slope, intercept, y2


def bea86_diameter_correction_func(
    xs,
    slope,
    intercept,
    y2
):
    """
    Calculate y value of the Been et al. (1986) diameter correction trend line.

    It should be noted that at x = 50 (i.e., diameter ratio = 50), Been et
    al. (1986) state that there are no longer any boundary effects and, thus,
    no diameter correction is needed.  This means that the slope of the trend
    line, at x = 50, is 0.

    Parameters
    ----------
    xs:  list of float | x data for plot
    slope:  float | slope of the Been et al. (1986) trend line
    intercept:  float | intercept of the Been et al. (1986) trend line
    y2:  float | y value at x = 50 (i.e., diameter ratio = 50) at which the
        Been et al. (1986) trend line stabilizes

    Returns
    -------
    ys:  list of float | y data for plot

    """
    # Make y data for the trend line.
    ys = [slope * x + intercept if x <= 50 else y2 for x in xs]

    return ys


def pore_pressure(
    depth,
    water_surface_loc,
    water_type='fresh'
):
    """
    Calculate pore pressure.

    Parameters
    ----------
    depth:  float | depth in m from the soil line (down is positive)
    water_surface_loc:  float | depth in m to the water surface from soil line
        (down is positive)
    water_type:  str | can be 'fresh' (9.81 kN/m^3) or 'sea' (10.03 kN/m^3)

    Returns
    -------
    pore_pressure: float | pore pressure in kPa

    """
    # Get the vertical distance between the water surface and the given depth.
    depth_water_surface = depth - water_surface_loc

    # Check if the given depth is above the water surface.
    if depth_water_surface < 0:
        raise ValueError('Given depth is above water surface')

    # Get the unit weight of water.
    if water_type == 'fresh':
        unit_weight = 9.81

    elif water_type == 'sea':
        unit_weight = 10.02

    else:
        raise ValueError("water_type must be either 'fresh' or 'sea'")

    pore_pressure = unit_weight * depth_water_surface

    return pore_pressure


def reset_cdb(cdb_loc):
    """
    Reset the CPT Database.

    Parameters
    ----------
    cdb_loc:  str | path to the CPT Database

    Returns
    -------
    None

    """
    # Get reset verification.
    verify = input('Reset CPT Database? (Y|n)') or 'y'

    # Check if verify is yes.
    if verify == 'y':
        cdb = {}

        with open(cdb_loc, 'wb') as out_file:
            pickle.dump(cdb, out_file)

        print('CPT Database reset...')

    # If verify is no...
    else:

        print('CPT Database not reset...')


def soil_layers(
    *points,
    bottom=-999999
):
    """
    Make soil layers.

    Parameters
    ----------
    points:  tup | tuple in the form (
            depth in m to top of soil layer,
            unit weight in kN/m^3
            )
    bottom:  flt | depth to bottom in m of the soil profile

    Returns
    -------
    layers:  dict | dictionary in the form {
            layer number: {
            'depth_range': (top, bottom),
            'unit_weight': unit_weight
            }}
    """
    # Get number of points
    num_points = len(points)

    # Make layers dictionary.
    layers = {}

    for count in range(len(points)):
        if count + 1 < num_points:
            layers[count + 1] = {
                'depth_range': (points[count][0], points[count + 1][0]),
                'unit_weight': points[count][1]
            }

        elif count + 1 == num_points:
            layers[count + 1] = {
                'depth_range': (points[count][0], bottom),
                'unit_weight': points[count][1]
            }

    return layers


def total_stress(
    depth,
    water_surface,
    layers,
    water_type='fresh'
):
    """
    Calculate total stress in kPa.

    Parameters
    ----------
    depth:  flt | depth in m (down is positive)
    water_surface:  flt | depth to water surface in m (down is positive)
    layers:  dict | dictionary in the form {
            layer number: {
                    'depth_range': (top, bottom),
                    'unit_weight': unit_weight
                    }
            }
    water_type:  str | can be 'fresh' (9.81 kN/m^3) or 'sea' (10.03 kN/m^3)

    Returns
    -------
    total_stress:  flt | total stress in kPa
    """
    # Get number of layers
    num_layers = len(layers.keys())

    layer = -999999
    total_stress = 0

    # Get layer.
    for count in range(len(layers.keys())):

        top, bottom = layers[count + 1]['depth_range']

        if depth > top and depth <= bottom:
            layer = count + 1

    if layer == -999999:
        raise ValueError('Depth exceeds soil profile')

    # Get unit weight of water.
    if water_type == 'fresh':
        unit_weight_water = 9.81

    elif water_type == 'sea':
        unit_weight_water = 10.02

    # Calculate vertical stress from water.
    if water_surface < 0:
        total_stress += abs(water_surface) * unit_weight_water

    # Calculate vertical stress from soil.
    for count in range(layer):

        top, bottom = layers[count + 1]['depth_range']
        unit_weight = layers[count + 1]['unit_weight']

        if count + 1 < layer:

            total_stress += (bottom - top) * unit_weight

        elif count + 1 == layer:
            total_stress += (depth - top) * unit_weight

    return total_stress

##################
def make_axes_cstate(
    **kwargs
):
    """
    Make axes with effective stress-normalized net tip resistance...

    Parameters
    ----------
    color:  str | can be any of the column headers from the soil, cptc, or
        chamber tables
    size:  str | can be any of the column headers from the soil, cptc, or
        chamber tables

    Other Parameters
    ----------------
    facecolor:  float | can be between 0 and 1
    c:  str | marker color (see PyPlot documentation)
    figsize:  tuple | figure size in inches in the form (width, height)

    Returns
    -------
    fig:  Matplotlib figure object
    ax:  Matplotlib axes object

    """
    # Update keyword arguments.
    parameters = {
        'facecolor': '0.95',
        'figsize': (6, 4),
        'xlim': None
    }

    parameters.update(kwargs)

    # initialize figure and axis objects
    fig, ax = plt.subplots(figsize=parameters['figsize'])

    # Set font.
    plt.rc(
        'text',
        usetex=True
    )
    plt.rc(
        'font',
        family='serif',
        size=12
    )

    # Set face color.
    ax.set_facecolor(parameters['facecolor'])

    # Axes
    ax.set_xscale('log')

    ax.set_xlabel(r'$\sigma_\mathrm{h0}^\prime$')
    ax.set_ylabel(r'$e$')

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    return fig, ax

def axes_deltaq(
    **kwargs
):
    """
    Make axes with effective stress-normalized net tip resistance...

    Parameters
    ----------
    color:  str | can be any of the column headers from the soil, cptc, or
        chamber tables
    size:  str | can be any of the column headers from the soil, cptc, or
        chamber tables

    Other Parameters
    ----------------
    facecolor:  float | can be between 0 and 1
    c:  str | marker color (see PyPlot documentation)
    figsize:  tuple | figure size in inches in the form (width, height)
    markeredgecolor:  str | marker edge color (see PyPlot documentation)
    markeredgewidth:  float | marker edge with in pixel
    markersize:  float | marker size in pixel

    Returns
    -------
    fig:  Matplotlib figure object
    ax:  Matplotlib axes object

    """
    # Update keyword arguments.
    parameters = {
        'facecolor': '0.95',
        'figsize': (6, 4),
        'xlim': None,
        #'major_locator': MultipleLocator(20),
        #'minor_locator': MultipleLocator(5)
    }

    parameters.update(kwargs)

    # initialize figure and axis objects
    fig, ax = plt.subplots(figsize=parameters['figsize'])

    # Set font.
    plt.rc(
        'text',
        usetex=True
    )
    plt.rc(
        'font',
        family='serif',
        size=12
    )

    # Set face color.
    ax.set_facecolor(parameters['facecolor'])

    # Set the axis labels.
    ax.set_xlabel(r'$f_\mathrm{s} / \sigma_\mathrm{v0}^\prime$')
    ax.set_ylabel(r'$Q_\mathrm{t}$')

    # Set the axis limits.
    #ax1.set_xlim(parameters['xlim'])

    # Set tick marks.
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    #ax1.yaxis.set_major_locator(parameters['major_locator'])
    #ax1.yaxis.set_minor_locator(parameters['minor_locator'])

    return fig, ax


def get_connection():
    """
    Get the connection to the Geotechnipy database.

    Returns
    -------
    connection:  sqlalchemy connection object | connection to the Calibration
        Chamber CPT database
    cptc:  sqlalchemy Table object | Calibation Chamber CPT database

    """
    # Make the engine to connect to the database.
    engine = create_engine('sqlite:///db/geotechnipy.sqlite')

    # Make the connection object.
    connection = engine.connect()

    return connection

label = {
    'compressibility': 'Compressibility, $\\lambda$',
    'curvature': 'Coefficient of curvature, $C_\\mathrm{c}$',
    'intercept': 'Intercept, $\\Gamma$',
    'oratio': 'Overconsolidation Ratio, $OCR$',
    'sparameter':  'State Parameter, $\\psi$',
    'uniformity': 'Coefficient of uniformity, $C_\\mathrm{u}$',
    'voidx': 'Maximum void ratio, $e_\\mathrm{max}$',
    'voidn': 'Minimum void ratio, $e_\\mathrm{min}$'
}

figsize = {
    'dr': (6, 4),
    'ds': (6, 6),
    'jr1c': (3.5, 2.33),
    'jr1.5c': (5.5, 3.66),
    'jr2c': (7.5, 5),
    'js1c': (3.5, 3.5)
}

size = {
    'dr': 12,
    'ds': 12,
    'jr1c': 8,
    'jr1.5c': 8,
    'jr2c': 8,
    'js1c': 8
}
