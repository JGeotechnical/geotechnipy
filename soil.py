from datetime import datetime
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy import and_, between
import pandas as pd
import warnings

class Soil():
    """
    Instantiate the Soils database class.

    This class enables the user to manipulate data relating to the engineering
    parameters of a soil in order to calculate and visualize further data also
    relating to the engineering parameters of the soil.

    """

    def __init__(
            self,
            sid=None,
            compressibility=None,
            intercept=None,
            deltaq=None,
            voidx=None,
            voidn=None,
            llimit=None,
            odllimit=None,
            plimit=None,
            pindex=None,
            sgravity=None,
            uscs=None,
            gradation=None,
            fines=None,
            sand=None,
            gravel=None,
            diameter10=None,
            diameter30=None,
            diameter50=None,
            diameter60=None,
            curvature=None,
            uniformity=None,
            shape=None,
            roundness=None,
            sphericity=None,
            organic=None,
    ):
        """
        Instantiate the Soils database class.

        The user should note that if the parameter is a list it should be
        specified in the form [min, max].

        Parameters
        ----------
        compressibility:  list | slope of the Critical State Line; although the
            geotechnical convention is for compressibility to be positive, it
            should be input as a negative number
        intercept:  list | intercept of the Critical State Line
        deltaq:  list | delta Q (see Saye et al. 2017)
        voidx:  list | maximum void ratio
        voidn:  list | minimum void ratio
        llimit:  list | liquid limit; negative number (e.g., -999999) indicates
            nonplastic
        plimit:  float | plastic limit; negative number (e.g., -999999)
            indicates nonplastic
        pindex:  float | plasticity index; negative number (e.g., -999999)
            indicates nonplastic
        sgravity:  list | specific gravity
        uscs:  list | Unified Soils Classification System classification; can
            be...
        diameter10:  list | particle diameter at 10% passing
        diameter30:  list | particle diameter at 30% passing
        diameter50:  list | particle diameter at 50% passing
        diameter60:  list | particle diameter at 60% passing
        gravel:  list | gravel content in decimal
        sand:  list | sand content in decimal
        fines:  list | fines content in decimal
        gradation:  Bool | True if sieve analysis results are attached; False
            otherwise
        organic:  Bool | True if the soil is organic, False otherwise
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
        roundness:  list | particle roundness
        sphericity:  list | particle sphericity

        """
        # Get the connection and soils table objects.
        connection, stable = get_db()

        # Filter the Soils Database.
        stmt = select([stable])

        ## Filter by soil ID.
        if sid:
            stmt = stmt.where(stable.columns.sid.in_((sid)))

        ## Filter by compressibility.
        if compressibility:
            stmt =stmt.where(
                between(
                    stable.columns.compressibility,
                    compressibility[0],
                    compressibility[1]
                )
            )

        ## Filter by CSL intercept.
        if intercept:
            stmt = stmt.where(
                between(
                    stable.columns.intercept,
                    intercept[0],
                    intercept[1]
                )
            )

        ## Filter by delta Q.
        if deltaq:
            stmt = stmt.where(
                between(
                    stable.columns.deltaq,
                    deltaq[0],
                    deltaq[1]
                )
            )

        ## Filter by maximum void ratio.
        if voidx:
            stmt = stmt.where(
                between(
                    stable.columns.voidx,
                    voidx[0],
                    voidx[1]
                )
            )

        ## Filter by minimum void ratio.
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

        ## Filter by plastic limit.
        if plimit:
            stmt = stmt.where(
                between(
                    stable.columns.plimit,
                    plimit[0],
                    plimit[1]
                )
            )

        ## Filter by plasticity index.
        if pindex:
            stmt = stmt.where(
                between(
                    stable.columns.pindex,
                    pindex[0],
                    pindex[1]
                )
            )

        ## Filter by specific gravity.
        if sgravity:
            stmt = stmt.where(
                between(
                    stable.columns.sgravity,
                    sgravity[0],
                    sgravity[1]
                )
            )

        ## Filter by USCS. <--Coming soon

        ## Filter by whether there is a gradation associated with the
        ## soil.
        if gradation:
            stmt = stmt.where(stable.columns.gradation.in_(gradation))

        ## Filter by particle diameter size.
        if diameter10:
            stmt = stmt.where(
                between(
                    stable.columns.diameter10,
                    diameter10[0],
                    diameter10[1]
                )
            )

        if diameter30:
            stmt = stmt.where(
                between(
                    stable.columns.diameter30,
                    diameter30[0],
                    diameter30[1]
                )
            )

        if diameter50:
            stmt = stmt.where(
                between(
                    stable.columns.diameter50,
                    diameter50[0],
                    diameter50[1]
                )
            )

        if diameter60:
            stmt = stmt.where(
                between(
                    stable.columns.diameter60,
                    diameter60[0],
                    diameter60[1]
                )
            )

        ## Filter by Coefficient of Curvature.
        if curvature:
            stmt = stmt.where(
                between(
                    stable.columns.curvature,
                    curvature[0],
                    curvature[1]
                )
            )

        ## Filter by Coefficient of Uniformity.
        if uniformity:
            stmt = stmt.where(
                between(
                    stable.columns.uniformity,
                    uniformity[0],
                    uniformity[1]
                )
            )

        ## Filter by percent gravel.
        if gravel:
            stmt = stmt.where(
                between(
                    stable.columns.gravel,
                    gravel[0],
                    gravel[1]
                )
            )

        ## Filter by percent sand.
        if sand:
            stmt = stmt.where(
                between(
                    stable.columns.sand,
                    sand[0],
                    sand[1]
                )
            )

        ## Filter by percent fines.
        if fines:
            stmt = stmt.where(
                between(
                    stable.columns.fines,
                    fines[0],
                    fines[1]
                )
            )

        ## Filter by roundness.
        if roundness:
            stmt = stmt.where(
                between(
                    stable.columns.roundness,
                    roundness[0],
                    roundness[1]
                )
            )

        ## Filter by particle shape.
        if shape:
            stmt = stmt.where(stable.columns.shape.in_(shape))

        ## Filter by sphericity.
        if sphericity:
            stmt = stmt.where(
                between(
                    stable.columns.sphericity,
                    sphericity[0],
                    sphericity[1]
                )
            )

        ## Filter by whether the soil is organic or not.
        if organic:
            stmt = stmt.where(stable.columns.organic.in_(organic))

        # Make the dataframe.
        self.__results = pd.read_sql(
            stmt,
            connection
        ).set_index('sid')

        self.__sids = self.__results.index.tolist()

    def __str__(self):
        """Print the selected soils"""
        return self.__results

    @property
    def results(self):
        return self.__results

    @property
    def voidx(self):
        """Gets maximum void ratio."""
        return self.__results.loc[:, 'voidx']

    @property
    def voidn(self):
        """Gets minimum void ratio."""
        return self.__results.loc[:, 'voidn']

    def gradation(
        self,
        legend=True,
        marker=True,
        sieves=(
                '3',
                '2',
                '1.5',
                '1',
                '3/4',
                '3/8',
                '4',
                '10',
                '20',
                '40',
                '60',
                '100',
                '140',
                '200'
        ),
        save=True,
        **kwargs
    ):
        """
        Plots the gradation of a soil.

        Parameters
        ----------
        legend:  bool | True to show legend; False otherwise
        marker:  bool | True to show markers; False otherwise
        sieves:  tuple | sieve sizes to annotate on the x axis (see table
            below)

            sieve size
            ----------
            '3'
            '2'
            '1.5'
            '1'
            '3/4'
            '3/8'
            '4'
            '10'
            '12'
            '14'
            '16'
            '18'
            '20'
            '25'
            '30'
            '35'
            '40'
            '45'
            '50'
            '60'
            '70'
            '80'
            '100'
            '120'
            '140'
            '170'
            '200'
        save:  bool | True to save; False otherwise

        kwargs
        ------
        See the axes_size_passing function.

        Returns
        -------
        None

        """
        # Get the plot axes.
        ax1, ax2 = axes_size_passing(
            sieves=sieves,
            **kwargs
        )

        for sid in self.__results.index.tolist():
            if self.__results.loc[sid]['gradation']:
                xs, ys = get_gradation(sid)

                ax1.plot(
                    xs,
                    ys,
                    color=self.__results.loc[sid]['scolor'],
                    label=self.__results.loc[sid]['sname'],
                    marker=(
                        self.__results.loc[sid]['smarker'] if marker is True else
                        None
                    ),
                    markeredgecolor='k',
                    markeredgewidth=0.15,
                    markersize = 6
                )

                xmin, xmax = ax1.get_xlim()
                ax2.set_xlim(xmin, xmax)

            else:
                warnings.warn(
                    'Sieve analysis data for {} not attached.'.format(sid)
                )

        if legend:
            ax1.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=3
                    )

        if save:
            time  = datetime.now()
            plt.savefig(
                'figs/gradation_{:%Y%m%d}_{:%H%M%S}.png'.format(time, time),
                bbox_inches='tight',
                dpi=600
            )
            plt.savefig(
                'figs/gradation_{:%Y%m%d}_{:%H%M%S}.pdf'.format(time, time),
                bbox_inches='tight'
            )

        plt.show()

    def plasticity(
        self,
        save=True
    ):
        """
        Returns the USCS classification for fine-grained soils.

        Parameters
        ----------
        save:  bool | True to save; False otherwise

        Returns
        -------
        None

        """
        # Get corners using A-Line equation.
        aLL = 20 + 7 / 0.73
        bPI = 0.73 * (50 - 20)
        cLL = 20 + 60 / 0.73

        # Initialize plasticity chart regions.
        clml_verts = [
            (4, 4),
            (7, 7),
            (aLL, 7),
            (25.5, 4),
            (4, 4)
        ]

        mlol_verts = [
            (0, 0),
            (4, 4),
            (25.5, 4),
            (50, bPI),
            (50, 0),
            (0, 0)
        ]

        mhoh_verts = [
            (50, 0),
            (50, bPI),
            (cLL, 60),
            (110, 60),
            (110, 0),
            (50, 0)
        ]

        choh_verts = [
            (50, bPI),
            (50, 50),
            (60, 60),
            (cLL, 60),
            (50, bPI)
        ]

        clol_verts = [
            (7, 7),
            (50, 50),
            (50, bPI),
            (aLL, 7),
            (7, 7)
        ]

        # Get centroids of each polygon.
        # Exclude the last point because it will skew the calculation since it
        # is a repeat of the first point.
        clml_centroid = centroid(clml_verts[:-1])
        mhoh_centroid = centroid(mhoh_verts[:-1])
        choh_centroid = centroid(choh_verts[:-1])
        clol_centroid = centroid(clol_verts[:-1])

        # Trim the mlol polygon because its placement interferes with the clml
        # polygon.
        mlol_centroid = centroid(mlol_verts[2:-1])

        # Initial codes list to draw polygons.
        five_points = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]

        six_points = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]

        clml = Path(clml_verts, five_points)
        mlol = Path(mlol_verts, six_points)
        mhoh = Path(mhoh_verts, six_points)
        choh = Path(choh_verts, five_points)
        clol = Path(clol_verts, five_points)

        # Set the font.
        plt.rc(
            'text',
            usetex=True
        )
        plt.rc(
            'font',
            family='serif',
            size=12
        )

        # Initialize figure and axis objects.
        fig1, ax1 = plt.subplots(figsize=(6, 4))

        # Add axis object to figure.
        fig1.add_subplot(ax1)

        # Add patch objects to figure.
        ax1.add_patch(patches.PathPatch(
            clml,
            alpha=0.1,
            edgecolor='k',
            facecolor='#1b9e77',
            linewidth=1
        ))

        ax1.add_patch(patches.PathPatch(
            mlol,
            alpha=0.1,
            edgecolor='k',
            facecolor='#d95f02',
            linewidth=1
        ))

        ax1.add_patch(patches.PathPatch(
            mhoh,
            alpha=0.1,
            edgecolor='k',
            facecolor='#7570b3',
            linewidth=1
        ))

        ax1.add_patch(patches.PathPatch(
            choh,
            alpha=0.1,
            edgecolor='k',
            facecolor='#e7298a',
            linewidth=1
        ))

        ax1.add_patch(patches.PathPatch(
            clol,
            alpha=0.1,
            edgecolor='k',
            facecolor='#66a61e',
            linewidth=1
        ))

        # Set text annotations.
        ax1.text(
            clml_centroid[0],
            clml_centroid[1],
            'CL-ML',
            horizontalalignment='center',
            verticalalignment='center'
        )

        ax1.text(
            mlol_centroid[0],
            mlol_centroid[1],
            'ML or OL',
            horizontalalignment='center',
            verticalalignment='center'
        )

        ax1.text(
            mhoh_centroid[0],
            mhoh_centroid[1],
            'MH or OH',
            horizontalalignment='center',
            verticalalignment='center'
        )

        ax1.text(
            choh_centroid[0],
            choh_centroid[1],
            'CH or OH',
            horizontalalignment='center',
            verticalalignment='center'
        )

        ax1.text(
            clol_centroid[0],
            clol_centroid[1],
            'CL or OL',
            horizontalalignment='center',
            verticalalignment='center'
        )

        # Set face color.
        ax1.set_facecolor('0.95')

        # Set minor ticks.
        ax1.minorticks_on()

        # Set axis ranges.
        ax1.set_xlim(0, 110)
        ax1.set_ylim(0, 60)

        # Set axis labels.
        ax1.set_xlabel('$LL$ (\%)')
        ax1.set_ylabel('$PI$')

        for index, row in self.__results.iterrows():

            # Plot the point.
            ax1.scatter(
                row['llimit'],
                row['pindex'],
                c=row['scolor'],
                edgecolors='black',
                label=row['sname'],
                linewidths=0.5,
                marker=row['smarker'],
                s=6**2,
                zorder=1
            )

        # Set legend
        ax1.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=3
        )

        if save:

            time  = datetime.now()

            plt.savefig(
                'figs/plasticity_{:%Y%m%d}_{:%H%M%S}.png'.format(time, time),
                bbox_inches='tight',
                dpi=600
            )
            plt.savefig(
                'figs/plasticity_{:%Y%m%d}_{:%H%M%S}.pdf'.format(time, time),
                bbox_inches='tight'
            )

        plt.show()

    def classify(
        self,
        overwrite=True
    ):

        '''
        Gets the USCS soil classification.

        Parameters
        ----------
        overwrite:  bool | True to update Soil table; False otherwise

        Returns
        -------
        symbols:  list | USCS symbol

        '''
        symbols = []

        for index, row in self.__results.iterrows():
            fines = row['fines']
            llimit = row['llimit']
            pindex = row['pindex']
            organic = row['organic']

            # Check if coarse-grained.
            if fines < 50:

                symbol = cgrained(
                    row['sand'],
                    row['curvature'],
                    row['uniformity'],
                    fines,
                    llimit,
                    pindex,
                    organic
                )

            else:
                symbol = fgrained(
                    llimit,
                    pindex,
                    organic
                )

            symbols.append(symbol)

            if overwrite is True:
                update_soil(
                    index,
                    uscs=symbol
                )

        return symbols


class SDbOld():
    """
    Instantiate the Soils database class.

    This class enables the user to manipulate data relating to the engineering
    parameters of a soil in order to calculate and visualize further data also
    relating to the engineering parameters of the soil.

    """

    def __init__(
            self,
            sdb_loc
    ):
        """Instantiate the Soils database class."""
        with open(sdb_loc, 'rb') as in_file:
            sdb = pickle.load(in_file)

        self.__dict__ = sdb

    def __str__(self):
        """Print the number of soils."""
        string = '{} soils loaded.'

        return string.format(len(self.__dict__))

    @property
    def ids(self):
        """List soil IDs."""
        return list(self.__dict__.keys())

    @property
    def sids(self):
        """List soil IDs."""
        stmt = select([self.soils.columns.sid])
        results = self.connection.execute(stmt).fetchall()
        sids = [result[0] for result in results]

        return sids

    @property
    def soils(self):
        """List soil names."""
        # Initialize empty list of soil names.
        soils = []

        for key in self.__dict__:

            # Get soil names.
            soils.append(self.__dict__[key]['soil_name'])

        return soils

    def cc(
        self,
        soil_id
    ):
        '''
        Get the coefficient of curvature.

        Parameters
        ----------
        soils:  str | name of soil

        Returns
        -------
        cc:  flt | coefficient of curvature
        '''''

        d60 = self.diameter(soil_id, 60)
        d30 = self.diameter(soil_id, 30)
        d10 = self.diameter(soil_id, 10)

        cc = d30 / (d10 * d60)

        return cc

    def classify(
        self,
        *soil_ids,
        sieve_nums=['3/8', '4', '10', '20', '40', '100', '200'],
        **kwargs
    ):

        '''
        Gets the USCS soil classification.

        Parameters
        ----------
        soil_ids:  str | soil IDs
        sieve_num: list | can be any of the following:
                          '3'
                          '2'
                          '1.5'
                          '1'
                          '3/4'
                          '3/8'
                          '4',
                          '10'
                          '12'
                          '14'
                          '16'
                          '18'
                          '20'
                          '25'
                          '30'
                          '35'
                          '40'
                          '45'
                          '50'
                          '60'
                          '70'
                          '80'
                          '100'
                          '120'
                          '140'
                          '170'
                          '200'

        Returns
        -------
        symbols:  list | USCS symbol
        '''

        symbols = []

        for soil_id in soil_ids:

            # Get fines content.
            fines_content = self.__dict__[soil_id]['fines_content']

            # Check if coarse-grained.
            if fines_content < 50:

                symbol = self.coarse_grained(soil_id)[0]

                # Check if dual symbol needed.
                if fines_content >= 5                and fines_content <= 12:
                    l3 = self.fine_grained(soil_id)[:1]

                    symbol = symbol + '-' + symbol[:1] + l3

                self.gradation(soil_id, sieve_nums = sieve_nums)

            else:
                symbol = self.fine_grained(soil_id)

                self.plasticity_chart(soil_id)

            symbols.append(symbol)

        return symbols

    def coarse_grained(
        self,
        *soil_ids
    ):

        '''
        Classifies coarse grained soils.

        Parameters
        ----------
        soil_names:  str | soil name

        Returns
        -------
        symbols:  list of str | USCS classification symbol
        '''

        symbols = []

        for soil_id in soil_ids:

            # Check if gravel or sand.
            if self.passing(soil_id, 4.75) < 50:
                l1 = 'G'

            else:
                l1 = 'S'

            # Get gradation coefficients.
            cc = self.cc(soil_id)
            cu = self.cu(soil_id)

            # Get gradation.
            if cc >= 4            and cu <= 1            and cu >= 3:
                l2 = 'W'

            else:
                l2 = 'P'

            symbols.append(l1 + l2)

        return symbols

    def fine_grained(
        self,
        *soil_ids
    ):

        '''
        Classifies fine grained soils.

        Parameters
        ----------
        soil_ids:  str | soil ID

        Returns
        -------
        symbols:  list of str | USCS classification symbol
        '''

        # Get corners of soil regions using A-Line equation.
        aLL = 20 + 7 / 0.73
        bPI = 0.73 * (50 - 20)
        cLL = 20 + 60 / 0.73

        # Initialize plasticity chart regions.
        clml = Path((
            [4, 4],
            [7, 7],
            [aLL, 7],
            [25.5, 4],
        ))

        mlol = Path((
            [0, 0],
            [4, 4],
            [25.5, 4],
            [50, bPI],
            [50, 0],
        ))

        mhoh = Path((
            [50, 0],
            [50, bPI],
            [cLL, 60],
            [110, 60],
            [110, 0],
        ))

        choh = Path((
            [50, bPI],
            [50, 50],
            [60, 60],
            [cLL, 60],
        ))

        clol = Path((
            [7, 7],
            [50, 50],
            [50, bPI],
            [aLL, 7],
        ))

        symbols = []

        for soil_id in soil_ids:
            # Get liquid limit and plasticity index.
            ll = self.__dict__[soil_id]['liquid_limit']
            pi = self.plasticity_index(soil_id)

            point = [[
                ll,
                pi
            ]]

            # Get the classification.
            if clml.contains_points(point)            and self.organic == False:
                symbol = 'CL-ML'
            elif mlol.contains_points(point)            and self.organic == False:
                symbol = 'ML'
            elif mhoh.contains_points(point)            and self.organic ==False:
                symbol = 'MH'
            elif choh.contains_points(point)            and self.organic == False:
                symbol = 'CH'
            elif clol.contains_points(point)            and self.organic == False:
                symbol = 'CL'
            elif mlol.contains_points(point)            and self.organic == True:
                symbol = 'OL'
            elif mhoh.contains_points(point)            and self.organic == True:
                symbol = 'OH'
            elif choh.contains_points(point)            and self.organic == True:
                symbol = 'OH'
            elif clol.contains_points(point)            and self.organic == True:
                symbol = 'OL'
            else:
                symbol = 'NP'

            symbols.append(symbol)

        return symbols

    def csl(
        self,
        *soils,
        **kwargs
    ):

        '''
        Plots the critical state line.

        Parameters
        ----------
        soils: str | name of soil

        Output
        ------
        Returns a figure.
        '''

        # Set kwargs.
        y = self.csl_slope * np.log10(2) + self.csl_intercept

        kwargs_ = {
            'figsize': (8.5, 7),
            'xy': (2, y),
            'xytext': (5, y * 1.15)
        }

        for key, value in kwargs.items():
            if key in kwargs_:
                kwargs_[key] = value

        # Get critical state database.
        csdb = get_csdb()

        # Initialize figure and axis objects.
        fig = plt.figure(1, figsize = kwargs_['figsize'])
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        # Change font.
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif', size = 12)

        # Set facecolor.
        ax1.set_facecolor('0.95')

        # Set x- and y-axis parameters.
        ax1.set_xscale('log')
        ax1.set_xlim(1, 1000)
        ax1.set_ylim(0, 1.1)
        ax2.set_ylim(ax1.get_ylim())

        ax1.minorticks_on()

        tick_labels = np.arange(-0.80, 1.20, 0.20)
        tick_locs = list(self.rd_2_vr(tick_labels))
        tick_labels = ['{:.2f}'.format(item) for item in tick_labels]
        ax2.set_yticks(tick_locs)
        ax2.set_yticklabels(tick_labels)

        ax1.set_xlabel(r'$\sigma^\prime_3$, kPa')
        ax1.set_ylabel(r'$e$')
        ax2.set_ylabel(r'$D_\mathrm{r}$, %')

        if self.csl_intercept is None        or self.csl_slope is None:
            pass
        elif self.csl_intercept is not None        and self.csl_intercept is not None:
            x = np.linspace(1, 1000, 1000)
            y = self.csl_slope * np.log10(x) + self.csl_intercept

        ax1.plot(
            x,
            y,
            color = 'k',
            linewidth =0.75,
            zorder = 3
        )

        # Plot the CSDb objects.
        for test_id, csdb_ in csdb.items():

            # Check sand name and if hidden.
            if csdb_['sand_name'] != self.name            or csdb_['hide'] == True            or csdb_['test_type'] != 'cu':
                continue

            # Plot initial state of undrained test.
            ax1.scatter(
                pd.Series(csdb_['stress_eff_mnr']).iloc[0],
                pd.Series(csdb_['voids_ratio']).iloc[0],
                alpha = 0.375,
                edgecolor = 'k',
                facecolor = csdb_['marker_color'],
                linewidth = 0.15,
                marker = csdb_['marker_type'],
                zorder = 2
            )

            # Get relative density.
            label_ =            '{} ($[D_\\mathrm{{r}}]_\\mathrm{{c}} = {:.2f}$)'            .format(csdb_['test_label'], pd.Series(csdb_['relative_density']).iloc[0])

            # Final state of undrained test.
            ax1.scatter(
                pd.Series(csdb_['stress_eff_mnr']).iloc[-1],
                pd.Series(csdb_['voids_ratio']).iloc[-1],
                edgecolor = 'k',
                facecolor = csdb_['marker_color'],
                label = label_,
                linewidth = 0.15,
                marker = csdb_['marker_type'],
                zorder = 2
            )

            # Line indicating change initial and final state of undrained test.
            ax1.plot(
                [
                    pd.Series(csdb_['stress_eff_mnr']).iloc[0],
                    pd.Series(csdb_['stress_eff_mnr']).iloc[-1]
                ],
                [
                    pd.Series(csdb_['voids_ratio']).iloc[0],
                    pd.Series(csdb_['voids_ratio']).iloc[-1]
                ],
                alpha = 0.375,
                color = csdb_['marker_color'],
                linewidth = 0.5,
                zorder = 1
            )

        # annotate the critical state line
        ax1.annotate(
            'CSL ($\\lambda = {}$, $\\Gamma = {}$)'.format(self.csl_slope, self.csl_intercept),
            xy = kwargs_['xy'],
            xytext = kwargs_['xytext'],
            size = 12,
            arrowprops = dict(
                arrowstyle = 'simple',
                fc = 'k',
                ec = 'k',
                connectionstyle = 'arc3, rad = 0.25'
        ))

        # set the legend
        ax1.legend(
            fontsize = 12,
            handlelength = 1,
            loc = 8,
            ncol = 2,
            scatterpoints = 1
        )

        plt.savefig('C:/Users/Joseph/Documents/Dissertation/Triaxial Tests/Critical State Plot/critical_state_line.pdf')
        plt.savefig('C:/Users/Joseph/Documents/Dissertation/Triaxial Tests/Critical State Plot/crtical_state_line.png', dpi=300)

        plt.show()

    def csl_slope(
            self,
            soil_id
            ):

        '''
        Gets the slope of the CSL line.

        Parameters
        ----------
        soil_id:  str | soil ID

        Returns
        -------
        csl_slope:  float | slope of the CSL line
        '''

        return self.__dict__[soil_id]['csl_slope']

    def cu(
        self,
        soil_id
    ):

        '''
        Gets the coefficient of uniformity.

        Parameters
        ----------
        soils:  str | name of soil

        Returns
        -------
        cu:  flt | coefficient of uniformity
        '''

        d60 = self.diameter(soil_id, 60)
        d10 = self.diameter(soil_id, 10)

        cu = d60 / d10

        return cu

    def diameter(
        self,
        soil_id,
        percent
    ):

        '''
        Gets soil particle diameter at a given percent passing.

        Parameters
        ----------
        percent:  int | Percent passing
        soil_name:  str | soil name

        Returns
        -------
        diameter:  flt | soil particle diameter in millimeters
        '''

        diameters, percents = self.__dict__[soil_id]['gradation_data']

        df = pd.DataFrame({
            'diameters': diameters,
            'percents': percents,
        })

        df.drop_duplicates(
            subset = 'percents',
            keep = 'last',
            inplace = True
        )

        df.sort_values(
            by = 'percents',
            inplace = True
        )

        # Interpolate soil particle diameter.
        diameter = np.interp(
            percent,
            df['percents'],
            df['diameters']
        )

        return diameter

    def gradation(
        self,
        *soil_ids,
        legend = False,
        sieve_nums = [
                '3',
                '2',
                '1.5',
                '1',
                '3/4',
                '3/8',
                '4',
                '10',
                '20',
                '40',
                '60',
                '100',
                '140',
                '200'
                ],
        **kwargs
    ):

        '''
        Plots the gradation of a coarse-grained soil.

        Parameters
        ----------
        soil_names:  str | soil name
        legend:  bool | False to not show legend; true to show legend
        sieve_nums:  list of str | can be any of the following:

            sieve_nums
            ----------
            '3'
            '2'
            '1.5'
            '1'
            '3/4'
            '3/8'
            '4'
            '10'
            '12'
            '14'
            '16'
            '18'
            '20'
            '25'
            '30'
            '35'
            '40'
            '45'
            '50'
            '60'
            '70'
            '80'
            '100'
            '120'
            '140'
            '170'
            '200'

        Returns
        -------
        Returns a figure showing the gradation of a soil.
        '''

        ### Keyword Arguments ###
        kwargs_ = {
        'set_facecolor': '0.95',
        'c': 'm',
        'figsize': (8.5, 7),
        'markeredgecolor': 'k',
        'markeredgewidth': 0.15,
        'markersize': 6
        }

        for key, value in kwargs.items():
            if key in kwargs_:
                kwargs_[key] = value

        # initialize figure and axis objects
        fig1, ax1 = plt.subplots(figsize = (6.5, 3.5))
        ax2 = ax1.twiny()

        # Set font.
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif', size = 12)

        # Set face color.
        ax1.set_facecolor('0.95')

        # Set the axis labels.
        ax1.set_xlabel('Particle Size, mm')
        ax1.set_ylabel('Percent Passing')

        ax2.set_xlabel('Sieve Size')

        # Set the axis attributes.
        ax1.set_xscale('log')
        ax2.set_xscale('log')

        # Set the axis limits.
        xrange = (10, 0.01)

        ax1.set_xlim(xrange)
        ax1.set_ylim(0, 100)

        ax2.set_xlim(xrange)

        # Set tick marks.
        major_loc = MultipleLocator(20)
        minor_loc = MultipleLocator(5)

        ax1.yaxis.set_major_locator(major_loc)
        ax1.yaxis.set_minor_locator(minor_loc)

        # Remove minor ticks on sieve number axis.
        ax2.get_xaxis().set_tick_params(
            which = 'minor',
            size = 0)
        ax2.get_xaxis().set_tick_params(
            which = 'minor',
            width = 0)

        # Set major ticks on sieve number axis.
        ## Make lists for tick locations and labels.
        locations = []
        labels = []

        ## Get tick locations and labels.
        for sieve in sieves:
            location, label = sieve_dict(sieve)

            locations.append(location)
            labels.append(label)
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels(
                tick_labels,
                rotation = 45,
                verticalalignment = 'bottom'
                )

        for soil_id in soil_ids:
            x, y = self.__dict__[soil_id]['gradation_data']

            ax1.plot(
                x,
                y,
                color = self.__dict__[soil_id]['soil_color'],
                label = self.__dict__[soil_id]['soil_name'],
                marker = self.__dict__[soil_id]['marker_type'],
                markeredgecolor = kwargs_['markeredgecolor'],
                markeredgewidth = kwargs_['markeredgewidth'],
                markersize = kwargs_['markersize']
            )

        # Check if legend wanted.
        if legend == True:

            # Set legend
            ax1.legend(
                    loc = 'upper left',
                    bbox_to_anchor = (1, 1)
                    )

        plt.show()

    def passing(
        self,
        soil_id,
        diameter,
    ):

        '''
        Gets the percent passing at a given diameter.

        Parameters
        ----------
        soil_name:  str | soil name
        diameter: flt | soil particle diameter in millimeters

        Returns
        -------
        percent:  flt | percent passing
        '''

        # Get gradation data.
        diameters, percents = self.__dict__[soil_id]['gradation_data']

        df = pd.DataFrame({
            'diameters': diameters,
            'percents': percents,
        })

        df.sort_values(
            by = 'percents',
            inplace = True
        )

        # Interpolate soil particle diameter.
        percent = np.interp(
            diameter,
            df['diameters'],
            df['percents']
        )

        return percent

    def plasticity_chart(
        self,
        *soil_ids,
        **kwargs
    ):

        '''
        Returns the USCS classification for fine-grained soils.

        Parameters
        ----------
        None

        Returns
        -------
        uscs:  str | USCS classification
        '''

        # Set keyword arguments.
        kwargs_ = {
        'set_facecolor': '0.95',
        'c': 'm',
        'figsize': (8.5, 7),
        'edgecolors': 'k',
        'linewidths': 0.15,
        's': 6**2,
        'size': 12,
        }

        for key, value in kwargs.items():
            if key in kwargs_:
                kwargs_[key] = value

        # Get corners using A-Line equation.
        aLL = 20 + 7 / 0.73
        bPI = 0.73 * (50 - 20)
        cLL = 20 + 60 / 0.73

        # Initialize plasticity chart regions.
        clml_verts = [
            (4, 4),
            (7, 7),
            (aLL, 7),
            (25.5, 4),
            (4, 4)
        ]

        mlol_verts = [
            (0, 0),
            (4, 4),
            (25.5, 4),
            (50, bPI),
            (50, 0),
            (0, 0)
        ]

        mhoh_verts = [
            (50, 0),
            (50, bPI),
            (cLL, 60),
            (110, 60),
            (110, 0),
            (50, 0)
        ]

        choh_verts = [
            (50, bPI),
            (50, 50),
            (60, 60),
            (cLL, 60),
            (50, bPI)
        ]

        clol_verts = [
            (7, 7),
            (50, 50),
            (50, bPI),
            (aLL, 7),
            (7, 7)
        ]

        # Get centroids of each polygon.
        # Exclude the last point because it will skew the calculation since it is a repeat of the first point.
        clml_centroid = centroid(clml_verts[:-1])
        mhoh_centroid = centroid(mhoh_verts[:-1])
        choh_centroid = centroid(choh_verts[:-1])
        clol_centroid = centroid(clol_verts[:-1])

        # Trim the mlol polygon because its placement interferes with the clml polygon.
        mlol_centroid = centroid(mlol_verts[2:-1])

        # Initial codes list to draw polygons.
        five_points = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]

        six_points = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]

        clml = Path(clml_verts, five_points)
        mlol = Path(mlol_verts, six_points)
        mhoh = Path(mhoh_verts, six_points)
        choh = Path(choh_verts, five_points)
        clol = Path(clol_verts, five_points)

        # Set the font.
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif', size = kwargs_['size'])

        # Initialize figure and axis objects.
        fig1, ax1 = plt.subplots(1, 1, figsize = kwargs_['figsize'])

        # Add axis object to figure.
        fig1.add_subplot(ax1)

        # Add patch objects to figure.
        ax1.add_patch(patches.PathPatch(
            clml,
            alpha = 0.25,
            edgecolor = 'k',
            facecolor = '#1b9e77',
            linewidth = 1
        ))

        ax1.add_patch(patches.PathPatch(
            mlol,
            alpha = 0.25,
            edgecolor = 'k',
            facecolor = '#d95f02',
            linewidth = 1
        ))

        ax1.add_patch(patches.PathPatch(
            mhoh,
            alpha = 0.25,
            edgecolor = 'k',
            facecolor = '#7570b3',
            linewidth = 1
        ))

        ax1.add_patch(patches.PathPatch(
            choh,
            alpha = 0.25,
            edgecolor = 'k',
            facecolor = '#e7298a',
            linewidth = 1
        ))

        ax1.add_patch(patches.PathPatch(
            clol,
            alpha = 0.25,
            edgecolor = 'k',
            facecolor = '#66a61e',
            linewidth = 1
        ))

        # Set text annotations.
        ax1.text(
            clml_centroid[0],
            clml_centroid[1],
            'CL-ML',
            horizontalalignment = 'center',
            verticalalignment = 'center'
        )

        ax1.text(
            mlol_centroid[0],
            mlol_centroid[1],
            'ML or OL',
            horizontalalignment = 'center',
            verticalalignment = 'center'
        )

        ax1.text(
            mhoh_centroid[0],
            mhoh_centroid[1],
            'MH or OH',
            horizontalalignment = 'center',
            verticalalignment = 'center'
        )

        ax1.text(
            choh_centroid[0],
            choh_centroid[1],
            'CH or OH',
            horizontalalignment = 'center',
            verticalalignment = 'center'
        )

        ax1.text(
            clol_centroid[0],
            clol_centroid[1],
            'CL or OL',
            horizontalalignment = 'center',
            verticalalignment = 'center'
        )

        # Set face color.
        ax1.set_facecolor(kwargs_['set_facecolor'])

        # Set minor ticks.
        ax1.minorticks_on()

        # Set axis ranges.
        ax1.set_xlim(0, 110)
        ax1.set_ylim(0, 60)

        # Set axis labels.
        ax1.set_xlabel(r'Plasticity Index, $PI$')
        ax1.set_ylabel(r'Liquid Limit, $LL$')

        symbols = []

        for soil_id in soil_ids:
            ll = self.__dict__[soil_id]['liquid_limit']
            pi = self.plasticity_index(soil_id)

            point = [[
                ll,
                pi
            ]]

            # Plot the point.
            ax1.scatter(
                ll,
                pi,
                c = self.__dict__[soil_id]['marker_color'],
                edgecolors = kwargs_['edgecolors'],
                label = self.__dict__[soil_id]['soil_name'],
                linewidths = kwargs_['linewidths'],
                marker = self.__dict__[soil_id]['marker_type'],
                s = kwargs_['s']
            )

            symbols.append(symbol)

        legend = ax1.legend(
            loc = 'upper left',
            bbox_to_anchor = (1, 1)
        )

        plt.show()

        return symbols

    def plasticity_index(
        self,
        soil_id
    ):

        '''
        Calculates plasticity index.
        '''

        plasticity_index = self.__dict__[soil_id]['liquid_limit'] - self.__dict__[soil_id]['plastic limit']

        return plasticity_index

    def specific_gravity(
        self,
        soil_id
    ):

        '''
        Gets the specific gravity of a soil.

        Parameters
        ----------
        soil_id:  str | soil ID

        Returns
        -------
        specific_gravity:  float | specific gravity
        '''

        return self.__dict__[soil_id]['specific_gravity']

    def vr_2_rd(
        self,
        soil_id,
        voids_ratio
    ):

        '''
        Converts void ratio to relative density.

        Parameters
        ----------
        soil_id:  str | soil ID
        voids_ratio:  float | voids ratio

        Output
        ------
        relative_density:  float | relative density
        '''

        relative_density = (self.__dict__[soil_id]['voids_ratio_max'] - voids_ratio)        / (self.__dict__[soil_id]['voids_ratio_max'] - self.__dict__[soil_id]['voids_ratio_min'])

        return relative_density

    def rd_2_vr(
        self,
        soil_id,
        relative_density
    ):

        '''
        Converts relative density to void ratio.

        Parameters
        ----------
        soil_name:  str | soil_name
        relative_density:  float | relative density

        Output
        ------
        voids_ratio:  float | voids ratio
        '''

        voids_ratio = self.__dict__[soil_id]['voids_ratio_max']- relative_density        * (self.__dict__[soil_id]['voids_ratio_max'] - self.__dict__[soil_id]['voids_ratio_min'])

        return voids_ratio

    def state_parameter(
        self,
        soil_id,
        voids_ratio,
        lateral_effective_stress
    ):

        '''
        Determines state parameter.

        Parameters
        ----------
        soil_id:  str | soil ID
        voids_ratio:  float | voids ratio
        vert_eff_stress:  float | vertical effective stress in kPa
        horz_eff_stress:  float | horizontal effective stress in kPa

        Returns
        -------
        state_parameter:  float | state parameter in decimal

        '''

        state_parameter = (voids_ratio - (
            self.__dict__[soil_id]['csl_slope'] *
            np.log10(lateral_effective_stress) +
            self.__dict__[soil_id]['csl_intercept']
            )
        )

        return state_parameter

    def sp_2_vr(
        self,
        lateral_effective_stress,
        soil_id,
        state_parameter
    ):
        """
        Calculate voids ratio.

        Parameters
        ----------
        lateral_effective_stress:  float | lateral effective stress in kPa
        soil_id:  str | soil ID
        state_parameter:  float | state parameter

        Returns
        -------
        voids_ratio:  float | voids ratio in decimal

        """
        voids_ratio = (state_parameter + (
            self.__dict__[soil_id]['csl_slope'] *
            np.log10(lateral_effective_stress) +
            self.__dict__[soil_id]['csl_intercept']
            )
        )

        return voids_ratio

    def soil(
            self,
            soil_id
            ):

        '''
        Gets the engineering parameters of a soil.

        Parameters
        ----------
        soil_id:  str | soil ID

        Returns
        -------
        soil_params:  dict | soil engineering parameters
        '''

        return self.__dict__[soil_id]

    def similitude(
        self,
        **kwargs
    ):
        '''
        Plots the critical state line.

        Output
        ------
        Returns a figure.
        '''

        # Set kwargs.
        y = self.csl_slope * np.log10(2) + self.csl_intercept

        kwargs_ = {
            'figsize': (8.5, 7),
            'xy': (2, y),
            'xytext': (5, y * 1.15)
        }

        for key, value in kwargs.items():
            if key in kwargs_:
                kwargs_[key] = value

        # Get critical state database.
        csdb = get_csdb()

        ##### FIGURE AND AXIS OBJECTS #####
        fig = plt.figure(1, figsize = kwargs_['figsize'])
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ##### FONT #####
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif', size = 12)

        ##### FACE COLOR #####
        ax1.set_facecolor('0.95')

        ##### X- AND Y-AXIS PARAMETERS #####
        ax1.set_xscale('log')
        ax1.set_xlim(1, 1000)
        ax1.set_ylim(0, 1.1)
        ax2.set_ylim(ax1.get_ylim())

        ##### TICKS #####
        ax1.minorticks_on()

        tick_labels = np.arange(-0.80, 1.20, 0.20)
        tick_labels = ['{:.2f}'.format(item) for item in tick_labels]
        tick_locs = list(self.rd_2_vr(tick_labels))

        ax2.set_yticks(tick_locs)
        ax2.set_yticklabels(tick_labels)

        ax1.set_xlabel(r'$\sigma^\prime_3$, kPa')
        ax1.set_ylabel(r'$e$')
        ax2.set_ylabel(r'$D_\mathrm{r}$, %')

        x = np.linspace(1, 1000, 1000)
        y = self.csl_slope * np.log10(x) + self.csl_intercept

        ax1.plot(
            x,
            y,
            color = 'k',
            linewidth = 0.75,
            zorder = 3
        )

        # Plot the CSDb objects.
        for test_id, csdb_ in csdb.items():

            # Check sand name and if hidden.
            if csdb_['sand_name'] != self.name\
            or csdb_['hide'] == True\
            or csdb_['test_type'] != 'cd':
                continue

            # Plot initial state of undrained test.
            ax1.scatter(
                pd.Series(csdb_['stress_eff_mnr']).iloc[0],
                pd.Series(csdb_['voids_ratio']).iloc[0],
                alpha = 0.375,
                edgecolor = 'k',
                facecolor = csdb_['marker_color'],
                linewidth = 0.15,
                marker = csdb_['marker_type'],
                zorder = 2
            )

            # Get relative density.
            label_ =            '{} ($[D_\\mathrm{{r}}]_\\mathrm{{c}} = {:.2f}$)'            .format(csdb_['test_label'], pd.Series(csdb_['relative_density']).iloc[0])

            # Final state of undrained test.
            ax1.scatter(
                pd.Series(csdb_['stress_eff_mnr']).iloc[-1],
                pd.Series(csdb_['voids_ratio']).iloc[-1],
                edgecolor = 'k',
                facecolor = csdb_['marker_color'],
                label = label_,
                linewidth = 0.15,
                marker = csdb_['marker_type'],
                zorder = 2
            )

            # Line indicating change initial and final state of undrained test.
            ax1.plot(
                [
                    pd.Series(csdb_['stress_eff_mnr']).iloc[0],
                    pd.Series(csdb_['stress_eff_mnr']).iloc[-1]
                ],
                [
                    pd.Series(csdb_['voids_ratio']).iloc[0],
                    pd.Series(csdb_['voids_ratio']).iloc[-1]
                ],
                alpha = 0.375,
                color = csdb_['marker_color'],
                linewidth = 0.5,
                zorder = 1
            )

        # annotate the critical state line
        ax1.annotate(
            'CSL ($\\lambda = {}$, $\\Gamma = {}$)'.format(self.csl_slope, self.csl_intercept),
            xy = kwargs_['xy'],
            xytext = kwargs_['xytext'],
            size = 12,
            arrowprops = dict(
                arrowstyle = 'simple',
                fc = 'k',
                ec = 'k',
                connectionstyle = 'arc3, rad = 0.25'
        ))

        # set the legend
        ax1.legend(
            fontsize = 12,
            handlelength = 1,
            loc = 8,
            ncol = 2,
            scatterpoints = 1
        )

        plt.savefig('C:/Users/Joseph/Documents/Dissertation/Triaxial Tests/Critical State Plot/critical_state_line.pdf')
        plt.savefig('C:/Users/Joseph/Documents/Dissertation/Triaxial Tests/Critical State Plot/crtical_state_line.png', dpi=300)

        plt.show()

def centroid(points):
    '''
    Calculates the centroid of a polygon.  Used to place labels in the
    plasticity_chart function.

    Parameters
    ----------
    points:  list of tuples | list in the form [(x1, y1), (x2, y2), ...]
    containing the vertices of a polygon

    Returns
    -------
    centroid: tuple | x and y coordinate of the centroid
    '''

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    centroid = (sum(x) / len(points), sum(y) / len(points))

    return centroid

def get_sdb(sdb_loc):
    '''
    Gets the Soil Database.

    Parameters
    ----------
    sdb_loc:  str | path to the Soil Database

    Returns
    -------
    sdb:  dict | Soil Database
    '''

    # Get Soil Database.
    with open(sdb_loc, 'rb') as in_file:
        sdb = pickle.load(in_file)

    return sdb


def get_db():
    """
    Get the Soils database.

    Returns
    -------
    connection:  sqlalchemy connection object | connection to the Soils
        database
    soils:  sqlalchemy Table object | Soils database

    """
    # Make the engine to connect to the database.
    #engine = create_engine('sqlite:///db/soils.sqlite')
    engine = create_engine('sqlite:///db/geotechnipy.sqlite')

    # Make the connection object.
    connection = engine.connect()

    # Make the metadata object.
    metadata = MetaData()

    # Reflect the table.
    soil = Table('soil', metadata, autoload=True, autoload_with=engine)

    return connection, soil


def create_soil(
    sname,
    srname,
    srdate,
    compressibility=None,
    intercept=None,
    deltaq=None,
    voidx=None,
    voidn=None,
    llimit=None,
    odllimit=None,
    plimit=None,
    pindex=None,
    sgravity=None,
    #uscs= <--coming soon
    sadata=False,
    diameter10=None,
    diameter30=None,
    diameter50=None,
    diameter60=None,
    curvature=None,
    uniformity=None,
    gravel=None,
    sand=None,
    fines=None,
    shape=None,
    roundness=None,
    sphericity=None,
    organic=False,
    scolor='black',
    smarker='o'
):
    """
    Creates a soil record in the Soils database.

    Parameters
    ----------
    sname:  str | soil name (e.g., Ottawa sand)
    srname:  str | name of the reference where the soil came from
    srdate:  float | date the reference was published

    Other Parameters
    ----------------
    compressibility:  float | slope of the Critical State Line; although the
        geotechnical convention is for compressibility to be positive, it
        should be input as a negative number here
    intercept:  float | intercept of the Critical State Line
    deltaq:  float | delta Q (see Saye et al. 2017)
    voidx:  float | maximum void ratio
    voidn:  float | minimum void ratio
    llimit:  float | liquid limit; negative number (e.g., -999999) indicates
        nonplastic
    odllimit:  float | liquid limit after oven drying
    plimit:  float | plastic limit; negative number (e.g., -999999) indicates
        nonplastic
    pindex:  float | plasticity index; negative number (e.g., -999999)
        indicates nonplastic
    sgravity:  float | specific gravity
    uscs:  int | Unified Soils Classification System classification; can be...
    diameter10:  float | particle diameter at 10% passing
    diameter30:  float | particle diameter at 30% passing
    diameter50:  float | particle diameter at 50% passing
    diameter60:  float | particle diameter at 60% passing
    gravel:  float | gravel content (percent)
    sand:  float | sand content (percent)
    fines:  float | fines content (percent)
    sadata:  Bool | True if sieve analysis data is attached in the form
        [[particle size], [percent passing]]; False otherwise
    organic:  Bool | True if the soil is organic (e.g., peat), False otherwise
    shape:  int | particle shape (see table below)
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
        --------------------------------------
    roundness:  float | particle roundness
    sphericity:  float | particle sphericity
    scolor:  str | soil color (for plotting purposes)
    smarker:  str | Pyplot marker style (see table below)
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
    part1 = ''.join(char for char in sname if char.isalnum()).lower()[:3]
    part2 = ''.join(char for char in srname if char.isalnum()).lower()[:3]

    # Make the soil ID.
    sid = part1 + '-' + part2 + '-' + str(srdate)[2:]

    # Combine the sieve analysis data.
    if sadata:
        gradation=True

        # Pickle the sieve analysis data.
        with open(
            'gradations/{}.p'.format(sid),
            'wb'
        ) as out_file:
            pickle.dump(
                sadata,
                out_file
            )

        # Create a dataframe with sizes and percents as columns.
        df = pd.DataFrame({
            'size': sadata[0],
            'percent': sadata[1],
        })

        # Drop duplicates from the percents columns.
        # Note:  If the gradation was digitized from literature, sometimes
        # there will be duplicate points.
        df.drop_duplicates(
            subset = 'percent',
            keep = 'last',
            inplace = True
        )

        # Sort the percent and size columns by percent in an ascending manner.
        df.sort_values(
            by = 'percent',
            inplace = True
        )
    else:
        gradation = False


    if (
        sadata and
        not fines
    ):
        fines = round(
            np.interp(
                0.075,
                df['size'],
                df['percent']
            ),
            1
        )
    elif (
        sadata and
        fines
    ):
        warnings.warn(
            'fines has been explicitly defined.  The explicitly defined value '
            'will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )
    if (
        sadata and
        not sand
    ):
        sand = round(
            np.interp(
                4.75,
                df['size'],
                df['percent']
            ),
            1
        )

        sand = sand - fines

    elif (
        sadata and
        sand
    ):
        warnings.warn(
            'sand has been explicitly defined.  The explicitly defined value '
            'will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )

    # Note:  Digitizing sometimes causes errors in the data.  In this
    # particular case, sometimes the percent of fines and sand add up to more
    # than 100%.
    # Proportionally adjust the percentage of fines and sands if necessary.
    if (
        fines and
        sand
    ):
        if fines + sand > 100:
            fines = round(
                fines / (fines + sand) * 100,
                1
            )
            sand = round(
                sand / (fines + sand) * 100,
                1
            )

    if (
        fines and
        sand and
        not gravel
    ):
        gravel =  round(
            100 - fines - sand,
            1
        )
    elif (
        fines and
        sand and
        gravel
    ):
        warnings.warn(
            'gravel has been explicitly defined.  The explicitly defined'
            'value will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )

    # Check if the percentages of each soil type add up to 100%
    if (
        fines and
        sand and
        gravel
    ):
        if gravel + sand + fines != 100:
            warnings.warn(
                'The percentages of each soil type do not add up to 100%.'
            )

    # Determine the soil particle diameters at 10, 30, 50, and 60 percent
    # passing.
    if (
        sadata and
        not diameter10
    ):

        diameter10 = np.interp(
            10,
            df['percent'],
            df['size']
        )
    elif (
        sadata and
        diameter10
    ):
        warnings.warn(
            'diameter10 has been explicitly defined.  The explicitly defined '
            'value will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )

    if (
        sadata and
        not diameter30
    ):
        diameter30 = np.interp(
            30,
            df['percent'],
            df['size']
        )
    elif(
        sadata and
        diameter30
    ):
        warnings.warn(
            'diameter30 has been explicitly defined.  The explicitly defined '
            'value will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )

    if (
        sadata and
        not diameter50
    ):
        diameter50 = np.interp(
            50,
            df['percent'],
            df['size']
        )
    elif(
        sadata and
        diameter50
    ):
        warnings.warn(
            'diameter50 has been explicitly defined.  The explicitly defined '
            'value will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )

    if (
        sadata and
        not diameter60
    ):
        diameter60 = np.interp(
            60,
            df['percent'],
            df['size']
        )
    elif (
        sadata and
        diameter60
    ):
        warnings.warn(
            'diameter60 has been explicitly defined.  The explicitly defined '
            'value will be used rather than the value that would have been '
            'interpolated from the sieve analysis data.'
        )

    # Calculate the Coefficient of Curvature.
    if (
        sadata and
        not curvature
    ):
        curvature =  diameter30 / (diameter10 * diameter60)
    elif (
        sadata and
        curvature
    ):
        warnings.warn(
            'curvature has been explicitly defined.  The explicitly defined '
            'value will be used rather than the value that would have been '
            'calculated.'
        )

    # Calculate the Coefficient of Uniformity.
    if (
        sadata and
        not uniformity
    ):
        uniformity =  diameter60 / diameter10
    elif (
        sadata and
        uniformity
    ):
        warnings.warn(
            'uniformity has been explicitly defined.  The explicitly defined '
            'value will be used rather than the value that would have been '
            'calculated.'
        )

    # Create a dictionary containing the columns and values.
    values = {
        'sid': sid,
        'sname': sname,
        'srname': srname,
        'srdate': srdate,
        'compressibility': compressibility,
        'intercept': intercept,
        'deltaq': deltaq,
        'voidx': voidx,
        'voidn': voidn,
        'llimit': llimit,
        'odllimit': odllimit,
        'plimit': plimit,
        'pindex': pindex,
        'gradation': gradation,
        'diameter10': diameter10,
        'diameter30': diameter30,
        'diameter50': diameter50,
        'diameter60': diameter60,
        'curvature': curvature,
        'uniformity': uniformity,
        'gravel': gravel,
        'sand': sand,
        'fines': fines,
        'roundness': roundness,
        'sgravity': sgravity,
        'shape': shape,
        'sphericity': sphericity,
        'organic': organic,
        'scolor': scolor,
        'smarker': smarker
    }

    # Get the connection and the soils table objects.
    connection, soils = get_db()

    # Update the database.
    connection.execute(insert(soils), [values])


def update_soil(
    sid,
    **kwargs
):
    """
    Updates 1 soil record in the Soils database.

    Parameters
    ----------
    sid:  str | soil ID
    args:  list | column and value to be inserted (see Other Parameters below)
      in the form [column, value]

    Other Parameters
    ----------------
    sname:  str | soil name (e.g., Ottawa sand)
    srname:  str | name of the reference where the soil came from
    srdate:  float | date the reference was published
    compressibility:  float | slope of the Critical State Line
    delta:  float | delta Q (see Saye et al. 2017)
    fines:  float | fines content in decimal
    gradation:  Bool | indicates if sieve analysis results are attached
    organic:  Bool | True if the soil is organic (e.g., peat), False otherwise
    intercept:  float | intercept of the Critical State Line
    llimit:  float | liquid limit
    plimit:  float | plastic limit
    pindex:  float | plasticity index
    round:  float | particle roundness
    sgrav:  float | specific gravity
    shape:  int | particle shape (see table below)
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
        | -999999 | not assigned             |
        --------------------------------------
    sphericity:  float | particle sphericity
    voidx:  float | maximum void ratio
    voidn:  float | minimum void ratio
    scolor:  str | soil color (for plotting purposes)
    smarker:  str | Pyplot marker style (see table below)
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
    connection, soils = get_db()

    # Create the query statement.
    stmt = update(soils)
    stmt = stmt.where(soils.columns.sid==sid)

    # Update the database.
    connection.execute(stmt, [kwargs])


def create_gradation(
    sid,
    psizes,
    ppassing,
    overwrite=True
):
    """
    Create a file containing the results of a sieve analysis.

    Parameters
    ----------
    sid:  str | soil ID
    sizes:  list | list of particle sizes in mm
    percents:  list | list of percents passing
    overwrite:  bool | True to update diameter10, diameter 30, diameter50, and
        diameter60 parameters in the Soils database; False otherwise

    """
    sadata = [psizes, ppassing]

    # Pickle the sieve analysis data.
    with open(
        'gradations/{}.p'.format(sid),
        'wb'
    ) as out_file:
        pickle.dump(
        sadata,
        out_file
    )

    # Create a dataframe with sizes and percents as columns.
    df = pd.DataFrame({
        'size': sizes,
        'percent': percents,
    })

    # Drop duplicates from the percents columns.
    # Note:  If the gradation was digitized from literature, sometimes there
    # will be duplicate points.
    df.drop_duplicates(
        subset = 'percent',
        keep = 'last',
        inplace = True
    )

    # Sort the percent and size columns by percent in an ascending manner.
    df.sort_values(
        by = 'percent',
        inplace = True
    )

    # Interpolate soil particle diameters at 10, 30, 50, and 60 percent
    # passing.
    diameter10 = np.interp(
        10,
        df['percent'],
        df['size']
    )

    # Interpolate soil particle diame# Interpolate soil particle diameter.ter.
    diameter30 = np.interp(
        30,
        df['percent'],
        df['size']
    )

    diameter50 = np.interp(
        50,
        df['percent'],
        df['size']
    )

    diameter60 = np.interp(
        60,
        df['percent'],
        df['size']
    )

    # Calculate the Coefficient of Curvature.
    curvature =  diameter30 / (diameter10 * diameter60)

    # Calculate the Coefficient of Uniformity.
    uniformity =  diameter60 / diameter10

    fines = np.interp(
        0.075,
        df['size'],
        df['percent']
    )

    sand = np.interp(
        4.75,
        df['size'],
        df['percent']
    )

    gravel =  100 - sand

    if overwrite:
        update_soil(
            sid,
            gradation=True,
            diameter10=diameter10,
            diameter30=diameter30,
            diameter50=diameter50,
            diameter60=diameter60,
            curvature=curvature,
            uniformity=uniformity,
            gravel=gravel,
            sand=sand,
            fines=fines,
        )

    else:
        update_soil(
            sid,
            gradation=True
        )


def get_gradation(
    sid
):
    """
    Get the file containing the results of a sieve analysis.

    Parameters
    ----------
    sid:  str | soil ID

    Returns
    -------
    grad:  list | list of the particles sizes and percents passing in the form
      [[particle sizes], [percents passing]]

    """
    # Unpickle the sieve analysis data.
    with open(
        'gradations/{}.p'.format(sid),
        'rb'
    ) as in_file:
        grad = pickle.load(in_file)

    return grad

def passing(
    diameter,
    sizes,
    percents
):
    """
    Gets the percent passing at a given diameter.

    Parameters
    ----------
    diameter: flt | particle size (mm)
    sizes:  flt | particles sizes
    percents:  float | percent passing

    Returns
    -------
    ppassing:  flt | percent passing

    """
    # Get gradation data.
    sizes, percents = get_gradation(sid)

    # Interpolate soil particle diameter.
    ppassing = np.interp(
        diameter,
        sizes,
        percents
    )

    return ppassing


def fgrained(
    llimit,
    pindex,
    organic
):

    '''
    Classifies fine grained soils.

    Parameters
    ----------
    llimit:  float | liquid limit
    pindex:  float | plasticity index
    organic:  bool | True if inorganic; false if organic

    Returns
    -------
    symbol:  str | USCS classification symbol

    '''
    # Get corners of soil regions using A-Line equation.
    aLL = 20 + 7 / 0.73
    bPI = 0.73 * (50 - 20)
    cLL = 20 + 60 / 0.73

    # Initialize plasticity chart regions.
    clml = Path((
        [4, 4],
        [7, 7],
        [aLL, 7],
        [25.5, 4],
    ))

    mlol = Path((
        [0, 0],
        [4, 4],
        [25.5, 4],
        [50, bPI],
        [50, 0],
    ))

    mhoh = Path((
        [50, 0],
        [50, bPI],
        [cLL, 60],
        [110, 60],
        [110, 0],
    ))

    choh = Path((
        [50, bPI],
        [50, 50],
        [60, 60],
        [cLL, 60],
    ))

    clol = Path((
        [7, 7],
        [50, 50],
        [50, bPI],
        [aLL, 7],
    ))

    point = [[
        llimit,
        pindex
    ]]

    # Get the classification.
    if (
        clol.contains_points(point) and
        organic is False
    ):
        return 'CL'

    if (
        clml.contains_points(point) and
        organic is False
    ):
        return 'CL-ML'

    if (
        mlol.contains_points(point) and
        organic == False
    ):
        return 'ML'

    if (
        choh.contains_points(point) and
        organic is False
    ):
        return 'CH'

    if (
        mhoh.contains_points(point) and
        organic is False
    ):
        return 'MH'

    if (
        mlol.contains_points(point) and
        organic is True
    ):
        return 'OL'

    if (
        mhoh.contains_points(point) and
        organic is True
    ):
        return 'OH'

    if (
        choh.contains_points(point) and
        organic is True
    ):
        return 'OL'

    if (
        clol.contains_points(point) and
        organic is True
    ):
        return 'OH'


def cgrained(
    sand,
    curvature,
    uniformity,
    fines,
    llimit,
    pindex,
    organic
):
    """
    Classifies coarse grained soils.

    Parameters
    ----------
    sand:  float | sand content (i.e., percent passing the number 4 sieve
        [4.75 mm])
    curvature:  float | coefficient of curvature
    uniformity:  float | coefficient of uniformity
    fines:  float | fines content (i.e., percent passing the number 200 sieve
        [0.075 mm])
    llimit:  float | liquid limit
    pindex:  float | plasticity index
    organic:  bool | True if inorganic; false if organic

    Returns
    -------
    symbol:  str | USCS classification symbol

    """
    if sand < 50:
        type = 'G'

        if (
            curvature >= 4 and
            1 <= uniformity <= 3
        ):
            gradation = 'W'

        else:
            gradation = 'P'
    else:
        type = 'S'

        if (
            curvature >= 6 and
            1 <= uniformity <= 3
        ):
            gradation = 'W'

        else:
            gradation = 'P'

    fsymbol = fgrained(
        llimit,
        pindex,
        organic
    )

    if fines < 5:
        symbol = type + gradation

    elif 5 <= fines <= 12:
        if (
            fsymbol == 'ML' or
            fsymbol == 'MH'
        ):
            symbol = type + gradation + '-' + type + 'M'

        elif (
            fsymbol == 'CL' or
            fsymbol == 'CH' or
            fsymbol == 'CL-ML'
        ):
            symbol = type + gradation + '-' + type + 'C'

    elif fines > 12:
        if (
            fsymbol == 'ML' or
            fsymbol == 'MH'
        ):
            symbol = type + 'M'

        elif (
            fsymbol == 'CL' or
            fsymbol == 'CH'
        ):
            symbol = type + 'C'

        elif fsymbol == 'CL-ML':
            symbol = type + 'M' + '-' + type + 'C'

    return symbol


def load_soil(
        sdb_loc,
        ref_proj_name,
        ref_proj_year,
        soil_name,
        soil_group,
        soil_color,
        soil_group_color,
        marker_type,
        csl_intercept=-999999,
        csl_slope=-999999,
        fines_content=-999999,
        gradation_data=[[-999999], [-999999]],
        organic=False,
        liquid_limit=-999999,
        plastic_limit=-999999,
        roundness=-999999,
        shape='not given',
        specific_gravity=-999999,
        sphericity=-999999,
        voids_ratio_max=-999999,
        voids_ratio_min=-999999
        ):

    '''
    Loads a soil into the Sand Database.

    Parameters
    ----------
    sdb_loc:  str | path to the Soil Database
    ref_proj_name:  str | reference or project name from where the data was
      obtained
    ref_proj_year:  int | four digit year (e.g., 2018)
    soil_name:  str | soil_name (e.g., Ottawa 40/60, Amauligak F-24 140/10)
    soil_group:  str | name of soil group (e.g., Ottawa, Amauligak F-24)
    soil_color:  str | marker color for this specific soil
    group_color:  str | marker color for this soil group
    marker_type:  str | marker type
    csl_intercept:  float | critical state line interept
    csl_slope:  float | critical state line slope
    fines_content:  float | fines content in percent
    gradation_data:  list | list of two lists in the form
      [[particle size in mm], [percent passing]]
    organic:  bool | True if the soil is organic (e.g., peat), False otherwise
    liquid_limit:  float | liquid limit in percent
    shape: str | can be 'angular', 'subangular', 'subrounded', 'rounded', or
      any combination of thereof (e.g., angular-subangular, subangular-rounded,
      etc.)
    specific_gravity:  float | specific gravity
    voids_ratio_max:  float | maximum voids ratio
    voids_ratio_min:  float | minimum voids ratio
    plastic_limit:  float | plastic limit in percent

    Returns
    -------
    None
    '''

    # Make soil ID.
    soil_id = (
            soil_name[:3].lower() +
            '-' +
            ref_proj_name[:3].lower() +
            '-' +
            str(ref_proj_year)[2:]
            )

    # Get Soil Database.
    with open(sdb_loc, 'rb') as in_file:
        sdb = pickle.load(in_file)

    sdb[soil_id] = {
        'csl_intercept': csl_intercept,
        'csl_slope': csl_slope,
        'fines_content': fines_content,
        'gradation_data': gradation_data,
        'organic': organic,
        'liquid_limit': liquid_limit,
        'soil_color': soil_color,
        'soil_group_color': soil_group_color,
        'marker_type': marker_type,
        'ref_proj_name': ref_proj_name,
        'ref_proj_year': ref_proj_year,
        'roundness': roundness,
        'shape': shape,
        'soil_group': soil_group,
        'soil_name': soil_name,
        'specific_gravity': specific_gravity,
        'voids_ratio_max': voids_ratio_max,
        'voids_ratio_min': voids_ratio_min,
        'plastic_limit': plastic_limit
    }

    # Save Soil Database.
    with open(sdb_loc, 'wb') as out_file:
        pickle.dump(sdb, out_file)

def mm2in(mm):
    """
    Convert millimeters to inches.

    Parameters
    ----------
    mm:  float | millimeters

    Returns
    -------
    inch:  float | inches

    """
    return mm * 0.0393700787


def axes_size_passing(
    sieves=(
            '3',
            '2',
            '1.5',
            '1',
            '3/4',
            '3/8',
            '4',
            '10',
            '20',
            '40',
            '60',
            '100',
            '140',
            '200'
            ),
    **kwargs
):
    """
    Make axes with particle size (x axis) versus percent passing (y axis)

    Parameters
    ----------
    sieves:  tuple | sieve sizes to annotate on the x axis (see table below)

        sieve size
        ----------
        '3'
        '2'
        '1.5'
        '1'
        '3/4'
        '3/8'
        '4'
        '10'
        '12'
        '14'
        '16'
        '18'
        '20'
        '25'
        '30'
        '35'
        '40'
        '45'
        '50'
        '60'
        '70'
        '80'
        '100'
        '120'
        '140'
        '170'
        '200'

    Other Parameters
    ----------------
    facecolor:  float | can be between 0 and 1
    c:  str | marker color (see PyPlot documentation)
    figsize:  tuple | figure size in inches in the form (width, height)
    markeredgecolor:  str | marker edge color (see PyPlot documentation)
    markeredgewidth:  float | marker edge with in pixel
    markersize:  float | marker size in pixel

    """
    # Update keyword arguments.
    parameters = {
        'facecolor': '0.95',
        'figsize': (6, 4),
        'xlim': (100, 0.01),
        'major_locator': MultipleLocator(20),
        'minor_locator': MultipleLocator(5)
    }

    parameters.update(kwargs)

    # initialize figure and axis objects
    fig1, ax1 = plt.subplots(figsize=parameters['figsize'])
    ax2 = ax1.twiny()

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
    ax1.set_facecolor(parameters['facecolor'])

    # Set the axis labels.
    ax1.set_xlabel('Particle Size, mm')
    ax1.set_ylabel('Percent Passing')
    ax2.set_xlabel('Sieve Size')

    # Set the x axes to log scale.
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # Set the axis limits.
    ax1.set_xlim(parameters['xlim'])
    ax1.set_ylim(0, 100)

    ax2.set_xlim(parameters['xlim'])

    # Set tick marks.
    ax1.yaxis.set_major_locator(parameters['major_locator'])
    ax1.yaxis.set_minor_locator(parameters['minor_locator'])

    # Remove minor ticks on sieve number axis.
    ax2.get_xaxis().set_tick_params(
        which='minor',
        size=0
    )
    ax2.get_xaxis().set_tick_params(
        which='minor',
        width=0
    )

    # Set major ticks on sieve number axis.
    ## Make lists for tick locations and labels.
    locations = []
    labels = []

    ## Get tick locations and labels.
    for sieve in sieves:
        location, label = sieve_dict[sieve]

        locations.append(location)
        labels.append(label)

    ## Set tick locations.
    ax2.set_xticks(locations)

    ## Set tick labels.
    ax2.set_xticklabels(
            labels,
            rotation=45,
            verticalalignment='bottom'
            )

    return ax1, ax2

def reset_sdb(sdb_loc):
    '''
    Resets the Soil Database.

    Parameters
    ----------
    sdb_loc:  str | path to the Soil Database

    Returns
    -------
    None

    '''
    sdb = {}

    with open(sdb_loc, 'wb') as out_file:
        pickle.dump(sdb, out_file)

shape_dict = {
    1: 'rounded',
    2: 'subrounded',
    3: 'subangular',
    4: 'angular',
    5: 'rounded to subrounded',
    6: 'rounded to subangular',
    7: 'rounded to angular',
    8: 'subrounded to subangular',
    9: 'subrounded to angular',
    10: 'subangular to angular',
    -999999: 'not assigned'
}


sieve_dict = {
    '3': (75, '3"'),
    '2': (50, '2"'),
    '1.5': (37.5, r'$1 \frac{1}{2}$"'),
    '1': (25, '1"'),
    '3/4': (19, r'$\frac{3}{4}$"'),
    '3/8': (9.5, r'$\frac{3}{8}$"'),
    '4': (4.75, '4'),
    '10': (2, '10'),
    '12': (1.7, '12'),
    '14': (1.4, '14'),
    '16': (1.18, '16'),
    '18': (1, '18'),
    '20': (0.85, '20'),
    '25': (0.71, '25'),
    '30': (0.6, '30'),
    '35': (0.5, '35'),
    '40': (0.425, '40'),
    '45': (0.355, '45'),
    '50': (0.3, '50'),
    '60': (0.25, '60'),
    '70': (0.212, '70'),
    '80': (0.18, '80'),
    '100': (0.15, '100'),
    '120': (0.125, '120'),
    '140': (0.106, '140'),
    '170': (0.09, '170'),
    '200': (0.075, '200')
}


uscs_dict = {
    'GW': 1,
    'GP': 2,
    'GW-GM': 3,
    'GW-GC': 4,
    'GP-GM': 5,
    'GP-GC': 6,
    'GM': 7,
    'GC': 8,
    'GC-GM': 9,
    'SW': 10,
    'SP': 11,

}
