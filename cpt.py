import ast
import pickle
import csv
from itertools import islice
from geotechnipy import soil
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

class CPT():
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

    def pen_resistance_vs_diam_ratio_viz(
            self,
            bcs=[1, 2, 3, 4, 5],
            csls_range=[-999999, 999999],
            soil_ids='all',
            sp_range=[-999999, 999999],
            save_loc='no'
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

        csls_range:  list of float | None or list in the form
            (min CSL slope, max CSL slope)

        sp_range:  list of float | None or list in the form
            (min state parameter, max state parameter)

        soil_ids:  list of str | 'all' or soil ID

        save_location:  str | 'yes' to save

        Returns
        -------
        None

        """
        # Get calibration chamber database.
        DF = self.get_data(
            bcs=bcs,
            csls_range=csls_range,
            sp_range=sp_range,
            soil_ids=soil_ids
        )

        if DF.empty:
            return None

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
        ax1.set_ylabel(r'$(q_\mathrm{c} - p) / p^\prime$')

        # Set the ticks.
        ax1.minorticks_on()
        ax1.tick_params(which='both', top=True, right=True)

        # Set x range.
        ax1.set_xlim([10, 70])

        # ---Make scatter plot--- #
        # If there are data points, make the scatter plot.
        ax1.scatter(
            DF['diameter_ratio'],
            DF['mesnn_tip_resistance'],
            alpha=0.5,
            c='#e41a1c',
            edgecolors='black',
            label='__nolegend__',
            linewidths=0.5,
            marker='o'
        )

        if save_loc != 'no':
            plt.savefig(
                save_loc + ".pdf",
                bbox_inches='tight',
                format='pdf'
            )

            plt.savefig(
                save_loc + ".png",
                bbox_inches='tight',
                format='png'
            )

        return fig1

    def diameter_ratio(
        self,
        cpt_id
    ):
        """
        Calculate the diameter ratio (i.e., chamber diameter / cone diameter).

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        diameter_ratio:  list of float | chamber diameter ratio

        """
        diameter_ratio = np.divide(
            np.array(self.__dict__[cpt_id]['chamber_diameter']),
            np.array(self.__dict__[cpt_id]['cone_diameter'])
        )

        return diameter_ratio

    def chamber_lateral_effective_stress(
        self,
        cpt_id
    ):
        """
        Calculate chamber lateral effective stress.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        chamber_lateral_stress:  float | lateral stress in kPa

        """
        chamber_lateral_effective_stress = (
            np.array(self.__dict__[cpt_id]['chamber_lateral_total_stress']) -
            np.array(self.__dict__[cpt_id]['hydrostatic_pore_pressures'][0])
        )

        return chamber_lateral_effective_stress

    def chamber_vertical_effective_stress(
        self,
        cpt_id
    ):
        """
        Calculate chamber total effective stress.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        cham_vert_eff_stress:  float | total effective stress in kPa

        """
        chamber_vertical_effective_stress = (
            np.array(self.__dict__[cpt_id]['chamber_vertical_total_stress']) -
            np.array(self.__dict__[cpt_id]['hydrostatic_pore_pressures'][0])
        )

        return chamber_vertical_effective_stress

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

    def osnnc_tip_resistance(
        self,
        cpt_id
    ):
        """
        Calculate overburden stress-normalized net corrected tip resistance.

        This function calculates the overbuden stress-normalized tip resistance
        for a calibration chamber test.  Thus, only a single value of tip
        resistance is returned.  This value is assumed to be representative of
        that particular soil under the conditions of the test (e.g., relative
        density, vertical effective stress, lateral effective stress, etc.).

        Parameters
        ----------
        cpt_id:  str | cpt name

        Returns
        -------
        osnnc_tip_resistance:  list of flt | overburden stress-normalized
            net corrected tip resistance in MPa

        """
        # Get tip resistance.
        tip_resistance = self.representative_value(
            cpt_id,
            self.__dict__[cpt_id]['tip_resistances']
        )

        # Convert from MPa to kPa.
        tip_resistance = tip_resistance * 1000

        # Get chamber vertical total stress.
        chamber_vertical_total_stress = (
            self.__dict__[cpt_id]['chamber_vertical_total_stress']
        )

        # Get chamber vertical effective stress.
        chamber_vertical_effective_stress = (
            self.chamber_vertical_effective_stress(cpt_id)
        )

        # Calculate the overburden stress-normalized net corrected tip stress.
        osnnc_tip_resistance = np.divide(
            tip_resistance - chamber_vertical_total_stress,
            chamber_vertical_effective_stress
        )

        return osnnc_tip_resistance

    def delta_q(
        self,
        cpt_id
    ):
        """
        Calculate delta_q.

        This function calculates the Delta q for a calibration chamber test.
        Thus, only a single value of Delta q is returned.  This value is
        assumed to be representative of that particular soil under the
        conditions of the test (e.g., relative density, vertical effective
        stress, lateral effective stress, etc.).

        Parameters
        ----------
        cpt_id:  str | CPT name

        Returns
        -------
        delta_q:  list of flt | delta_q

        """
        delta_q = np.divide(
            self.osnnc_tip_resistance(cpt_id) + 10,
            self.esn_sleeve_friction(cpt_id) + 0.67
        )

        return delta_q

    def mean_total_stress(
            self,
            cpt_id
            ):
        """
        Calculate mean total stress.

        The mean total stress, P, is defined as

        P = (sigma_1 + sigma_3 * 2) / 3

        where

        sigma_1 = major axis effective stress
        sigma_3 = minor axis effective stress

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        mean_total_stress:  list of float | mean total stress in kPa

        """
        # Get chamber total stress.
        chamber_vertical_total_stress = (
                self.__dict__[cpt_id]['chamber_vertical_total_stress']
                )

        # Get chamber lateral stress.
        chamber_lateral_total_stress = (
                self.__dict__[cpt_id]['chamber_lateral_total_stress']
                )

        # Calculate mean total stress.
        mean_total_stress = (
            chamber_vertical_total_stress +
            chamber_lateral_total_stress * 2
        ) / 3

        return mean_total_stress

    def mean_effective_stress(
        self,
        cpt_id
    ):
        """
        Calculate mean effective stress.

        The mean effective stress, P', is defined as

        P' = (sigma_1' + sigma_3' * 2) / 3

        where

        sigma_1'    = major axis effective stress
        sigma_3'    = minor axis effective stress

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        mean_effective_stress:  list of float | mean effective stress in kPa

        """
        # Get chamber vertical effective stress.
        chamber_vertical_effective_stress = (
            self.chamber_vertical_effective_stress(cpt_id)
        )

        # Get chamber lateral effective stress.
        chamber_lateral_effective_stress = (
            self.chamber_lateral_effective_stress(cpt_id)
        )

        # Calculate mean effective stress.
        mean_effective_stress = (
            chamber_vertical_effective_stress +
            chamber_lateral_effective_stress * 2
        ) / 3

        return mean_effective_stress

    def mesnn_tip_resistance(
        self,
        cpt_id
    ):
        """
        Calculate mean-effective-stress-normalized net tip resistance.

        The mean-effective-stress-normalized tip resistance, [q_c]_P', is
        defined by the equation

        [q_c]_P' = (q_c - P) / P'

        where

        q_c     = tip Resistance
        P       = (sigma_1 + sigma_3 * 2) / 3
        P'      = (sigma_1' + sigma_3' * 2) / 3

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        mesnn_tip_resistance:  list of float | normalized,
            corrected tip resistance

        """
        # Get tip resistance and convert from MPa to kPa.
        tip_resistance = np.array(
            self.__dict__[cpt_id]['tip_resistances']
            ) * 1000

        # Get total stress.
        mean_total_stress = np.array(self.mean_total_stress(cpt_id))

        # Get effective stress.
        mean_effective_stress = np.array(self.mean_effective_stress(cpt_id))

        # Calculate the mean-effective-stres-normalized, corrected tip
        # resistance.
        with np.errstate(divide='ignore', invalid='ignore'):
            mesnn_tip_resistances = np.divide(
                (tip_resistance - mean_total_stress),
                mean_effective_stress
            )

            # Replace infinite values, if any, with 0.
            mesnn_tip_resistances[
                mesnn_tip_resistances == np.inf
            ] = 0

            # Replace nans, if any, with 0.
            mesnn_tip_resistances = np.nan_to_num(
                mesnn_tip_resistances
            )

        # Calculate the representative value of the mean stress
        # normalized tip resistance.
        mesnn_tip_resistance = self.representative_value(
            cpt_id,
            mesnn_tip_resistances
        )

        return mesnn_tip_resistance

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

    def oc_coefs(
        self,
        cpt_id
    ):
        """
        Get the overburden stress correction coefficients.

        The equation for the C trendline came from the overburden correction
        correction analysis in Gamez (201X).

        Parameters
        ----------
        cpt_id:  str | Cone penetration test ID

        Returns
        -------
        c:  float | overburden coefficient
        n:  float | overburden coefficient
        """
        # Get relative density.
        relative_density = self.__dict__[cpt_id]['relative_density']

        # Get state parameter.
        state_parameter = self.__dict__[cpt_id]['state_parameter']

        # Calculate C slope.
        cslope = (
            0.83824581969226064 * (relative_density * 100)
            + 31.989666113947862
        )

        # Calculate C intercept.
        cintercept = (
            0.6996728047055456 * (relative_density * 100) + 8.6915827719759733
        )

        # Calculate C coefficient.
        c = cslope * state_parameter + cintercept

        # Calculate n slope.
        nslope = (
            0.8458638664361533 * np.log(relative_density * 100)
            - 3.9710599896226606
        )

        # Calculate n coefficient.
        nintercept = (
            -0.0020749193312107677 * (relative_density * 100)
            + 0.75795629551097621
        )

        # Calculate n coefficient.
        n = nslope * state_parameter + nintercept

        return (c, n)

    def oc_factor(
        self,
        cpt_id,
        atmospheric_pressure=100.
    ):
        """
        Calculate the overburden correction factor.

        Parameters
        ----------
        cpt_id:  str | Cone penetrtion test ID
        atmospheric_pressure:  float | atmospheric pressure; generally
          estimated as 100 kPa

        Returns
        -------
        overburden_corrected_tip_resistance:  float | estimated tip resistance
        at 1 atmosphere of pressure.

        """
        # Get the overburden coefficients (i.e., C and n).
        c, n = self.oc_coefs(cpt_id)

        # Get the vertical effective stress.
        if self.__dict__[cpt_id]['cpt_type'] == 'chamber':
            vertical_effective_stress = (
                self.chamber_vertical_effective_stress(cpt_id)
            )

        elif self.__dict__[cpt_id]['cpt_type'] == 'field':
            vertical_effective_stress = (
                self.__dict__[cpt_id]['vertical_effective_stress']
            )

        # Calculate the overburden stress correction factor (i.e., Cn).
        oc_factor = (atmospheric_pressure / vertical_effective_stress)**n

        if oc_factor > 1.7:
            oc_factor = 1.7

        return oc_factor

    def esn_sleeve_friction(
        self,
        cpt_id
    ):
        """
        Calculate the effective-stress-normalized sleeve friction.

        Parameters
        ----------
        cpt_id:  str | CPT ID

        Returns
        -------
        norm_slv_fric:  list of float | sleeve friction normalized with
        effective stress
        """
        # Get sleeve friction.
        sleeve_friction = self.representative_value(
            cpt_id,
            self.__dict__[cpt_id]['sleeve_frictions']
        )

        # Get chamber vertical effective stress.
        chamber_vertical_effective_stress = (
            self.chamber_vertical_effective_stress(cpt_id)
        )

        esn_sleeve_friction = np.divide(
            np.array(sleeve_friction),
            np.array(chamber_vertical_effective_stress)
        )

        return esn_sleeve_friction

    def representative_value(
            self,
            cpt_id,
            datas
            ):
        """
        Gets the representative value for a data series (e.g., tip resistance,
        skin friction, lateral stress, etc.) in a calibration chamber test.
        Gamez (201X) found that the values of a data series in a calibration
        chamber test stabilize at a depth of 4 times the penetrometer below the
        top of the chamber and a height of 8 times the penetrometer diameter
        above the bottom of the chamber.

        Parameters
        ----------
        data:  list of float | data from a calibration chamber test (e.g., tip
            resistance, skin friction, lateral stress, etc.)

        Returns
        -------
        representative_value:  float | average of values below 4 x dc and above
            8 x dc

        """
        # Get CPT type.
        cpt_type = self.__dict__[cpt_id]['cpt_type']

        # Check if sounding is from a field test.
        if cpt_type == 'field':

            raise TypeError(
                    'representative_value function can only be used with'
                    'calibration chamber tests...'
                    )

        # Check if sounding is from a calibration chamber test.
        elif cpt_type == 'chamber':

            # Get cone diameter and convert from cm to m.
            cone_diameter = self.__dict__[cpt_id]['cone_diameter'] / 100

            # Get chamber height and convert from cm to m.
            chamber_height = self.__dict__[cpt_id]['chamber_height'] / 100

            # Calculate the upper boundary
            upper = 4 * cone_diameter

            # Calculate the lower boundary
            lower = chamber_height - 8 * cone_diameter

            # Get depths.
            depths = self.__dict__[cpt_id]['depths']

            data_in_range = []

            for data, depth in zip(datas, depths):

                # Check if depth is below below 4 x dc and above 8 x dc.
                if upper <= depth <= lower:

                    # Append data to list.
                    data_in_range.append(data)

            # Get the mean.
            average = np.mean(data_in_range)

            return average


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
    cdb = {}

    with open(cdb_loc, 'wb') as out_file:
        pickle.dump(cdb, out_file)

    print('CPT Database reset...')


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
