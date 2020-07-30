import ast
from datetime import datetime
import pickle
import csv
from itertools import islice
from geotechnipy import soil
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy import and_, between

class DCP():
    """Instantiate the Dynamic Cone Penetrometer base class."""

    def __init__(
        self,
        *did
    ):
        """
        Instatiate the Dynamic Cone Penetrometer base class.

        Parameters
        ----------
        did:  str | DCP ID

        """
        # Get the connection and cptc sqlalchemy objects.
        connection, table = get_table()

        # Filter the Soils Database.
        stmt = select([table])

        ## Filter by soil ID.
        if did:
            stmt = stmt.where(table.columns.did.in_((did)))

        ## Make the dataframe.
        self.__results = pd.read_sql(
            stmt,
            connection
        ).set_index('did')

    @property
    def table(self):
        return self.__results

    def plot(
        self,
        *layers,
        uscs='default',
        save=True,
        **kwargs
    ):
        """
        Make a plot of PI and CBR versus depth.

        Parameters
        ----------
        did:  str | DCP ID
        layers:  tuple | depth and representative value in the form (
                depth,
                blow count representative value (can be int or None),
                penetration index representative value (can be int or None),
                California Bearing Ratio representative value (can be int or
                    None)
            )
        uscs:  str | Unified Soil Classification System symbol of the soil
        save:  bool | True to save; False otherwise

        Returns
        -------
        ax1:  Pyplot axis object | blow count axis (mm)
        ax1i:  Pyplot axis object | blow count axis (in)
        ax2:  Pyplot axis object | penetration index axis (mm)
        ax2i:  Pyplot axis object | penetration index axis (in)
        ax3:  Pyplot axis object | California Bearing Ratio axis (mm)
        ax3i:  Pyplot axis object | California Bearing Ratio axis (in)

        """
        defaults = {
            'ax1_xlim': [0, 100],
            'ax2_xlim': [1, 100],
            'ax3_xlim': [1, 100],
            'ylim': [1000, 0]
        }

        defaults.update(kwargs)

        # Make figure and axis objects.
        fig1, axs = plt.subplots(
            figsize=(7.5, 9),
            ncols=3,
            sharey=True
        )

        ax1, ax2, ax3 = axs

        ax1i = ax1.twinx()
        ax2i = ax2.twinx()
        ax3i = ax3.twinx()

        # Set font.
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)

        # Set face color.
        ax1.set_facecolor('0.95')
        ax2.set_facecolor('0.95')
        ax3.set_facecolor('0.95')

        # Set the axis labels.
        ax1.set_xlabel('Blows')
        ax1.set_ylabel('Depth (mm)')
        ax2.set_xlabel('PI (mm/blow)')
        ax3.set_xlabel('CBR')
        ax3i.set_ylabel('Depth (in)')

        # Set the axis ranges.
        ax1.set_xlim(defaults['ax1_xlim'])
        ax1.set_ylim(defaults['ylim'])
        ax1i.set_ylim(defaults['ylim'])
        ax2.set_xlim(defaults['ax2_xlim'])
        ax2.set_ylim(defaults['ylim'])
        ax2i.set_ylim(defaults['ylim'])
        ax3.set_xlim(defaults['ax3_xlim'])
        ax3.set_ylim(defaults['ylim'])
        ax3i.set_ylim(defaults['ylim'])

        # Set the axis scales.
        ax2.set_xscale('log')
        ax3.set_xscale('log')

        # Set the tick locations.
        ax1.xaxis.tick_top()
        ax2.xaxis.tick_top()
        ax3.xaxis.tick_top()

        # Set the tick positions.
        ax1.xaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax3.xaxis.set_ticks_position('both')

        # Set the major ticks.
        mms = np.linspace(0, 1000, 11)
        inches = np.linspace(0, 40, 9)
        inches2mms = [
            int(
                inch / 0.03937008
            ) for inch in inches if inch  / 0.03937008 < 1000
        ]

        ax1.set_yticks(mms)
        ax1i.set_yticks(inches2mms)
        ax3.set_yticks(mms)
        ax2i.set_yticks(inches2mms)
        ax3.set_yticks(mms)
        ax3i.set_yticks(inches2mms)

        # Set the minor ticks.
        ax1.minorticks_on()
        ax1i.minorticks_on()
        ax2.minorticks_on()
        ax2i.minorticks_on()
        ax3.minorticks_on()
        ax3i.minorticks_on()

        # Set axis labels.
        ax1.xaxis.set_label_position('top')
        ax2.xaxis.set_label_position('top')
        ax3.xaxis.set_label_position('top')

        ax1i.tick_params(
            labelright=False
        )
        ax2i.tick_params(
            labelleft=False,
            labelright=False
        )
        ax3i.set_yticklabels(inches)

        # Get the results.
        for index, row in self.__results.iterrows():
            bcounts, depths = get_bcounts(index)

            # Remove the 0 reading from the blow counts and duplicate the next
            # reading (for visualization purposes).
            bcounts_ = bcounts[1:]
            bcounts_.insert(0, bcounts_[0])

            # Plot blow count.
            ax1.step(
                bcounts_,
                depths,
                color=row['dcolor'],
                where='post'
            )

            # Get the differential depth between each reading.
            differentials = np.diff(depths).tolist()

            # Calculate penetration index.
            pindices = [
                pindex(
                    bcount,
                    differential
                ) for bcount, differential in zip(bcounts[1:], differentials)
            ]

            # Add a duplicate of the first penetration index (for visualization
            # purposes).
            pindices.insert(0, pindices[0])

            # Plot penetration index.
            ax2.step(
                pindices,
                depths,
                color=row['dcolor'],
                where='post'
            )

            if uscs == 'default':
                cbratios = [cbr(pindex) for pindex in pindices]
            elif uscs == 'ch':
                cbratios = [cbr_ch(pindex) for pindex in pindices]
            elif uccs == 'cl':
                cbratios = [
                    cbr_cl(pindex) if cbr_cl(pindex) < 10 else cbr(pindex) \
                    for pindex in pindices
                ]

            # Plot CBR.
            ax3.step(
                cbratios,
                depths,
                color=row['dcolor'],
                where='post'
            )

        plt.tight_layout()

        if save is True:

            time  = datetime.now()

            plt.savefig(
                'figs/dcp_results_{:%Y%m%d}_{:%H%M%S}.png'.format(time, time),
                dpi=600
            )
            plt.savefig(
                'figs/dcp_results_{:%Y%m%d}_{:%H%M%S}.pdf'.format(time, time)
            )

        plt.show()

        return ax1, ax1i, ax2, ax2i, ax3, ax3i


def pindex(
    bcount,
    depth
):
    """
    Calculate penetration index from blow count.

    Parameters
    ----------
    bcount:  int | blow count
    depth:  float | penetration depth (mm)

    Returns
    -------
    pindex:  float | penetration index (mm/blow)

    """
    if bcount == 0:
        pindex = 0

    else:
        pindex = depth / bcount

    return pindex

def cbr(
    pindex
):
    """
    Calculate CBR from penetration index for soils other than CH or CL.

    Parameters
    ----------
    pindex:  float | penetration index

    Returns
    -------
    cbratio:  float | California Bearing Ratio

    """
    if pindex == 0:
        cbratio = 0
    else:
        cbratio = 292 / (pindex)**1.12

    return cbratio

def cbr_ch(
    pindex
):
    """
    Calculate CBR from penetration index for CH soils.

    Parameters
    ----------
    pindex:  float | penetration index

    Returns
    -------
    cbratio:  float | California Bearing Ratio

    """
    if pindex == 0:
        cbratio = 0
    else:
        cbratio = 1 / (0.002871 * pindex)

    return cbratio


def cbr_cl(
    pindex
):
    """
    Calculate CBR from penetration index for CL soils.

    Parameters
    ----------
    pindex:  float | penetration index

    Returns
    -------
    cbratio:  float | California Bearing Ratio

    """
    if pindex == 0:
        cbratio = 0
    else:
        cbratio = 1 / (0.002871 * pindex)**2

    return cbratio


def create_dcp(
    dname,
    drname,
    drdate,
    dmass=8.0,
    dcolor='black',
    dmarker='o'
):
    """
    Create a record in the DCP database.

    Parameters
    ----------
    dname:  str | DCP test name
    drname:  str | name of the reference where the test came
    drdate:  float | date the reference was published
    bcount:  bool | False if there is no blow count data attached; True
        otherwise
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
    # Remove special characters, spaces, etc. in sname and srname.
    part1 = ''.join(char for char in dname if char.isalnum()).lower()
    part2 = ''.join(char for char in drname if char.isalnum()).lower()[:3]

    # Make the soil ID.
    did = part1 + '-' + part2 + '-' + str(drdate)[2:]

    # Create a dictionary containing the columns and values.
    values = {
        'did': did,
        'dname': dname,
        'drname': drname,
        'drdate': drdate,
        'bcount': False,
        'dmass': dmass,
        'dcolor': dcolor,
        'dmarker': dmarker
    }

    # Get the connection and the soils table objects.
    connection, table = get_table()

    # Update the database.
    connection.execute(insert(table), [values])


def update_table(
    did,
    **kwargs
):
    """
    Updates 1 soil record in the DCP database.

    Parameters
    ----------
    did:  str | DCP ID

    Other Parameters
    ----------------
    dname:  str | DCP test name
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
    # Get the connection and soils table objects.
    connection, table = get_table()

    # Create the query statement.
    stmt = update(table)
    stmt = stmt.where(table.columns.did==did)

    # Update the database.
    connection.execute(stmt, [kwargs])


def get_table():
    """
    Get the DCP database.

    Returns
    -------
    connection:  sqlalchemy connection object | connection to the Calibration
        Chamber CPT database
    table:  sqlalchemy Table object | DCP database

    """
    # Make the engine to connect to the database.
    engine = create_engine('sqlite:///db/geotechnipy.sqlite')

    # Make the connection object.
    connection = engine.connect()

    # Make the metadata object.
    metadata = MetaData()

    # Reflect the table.
    table = Table('dcp', metadata, autoload=True, autoload_with=engine)

    return connection, table


def add_layer(
    *axes
):
    """
    Add soil layers to the results plot.

    This function can only be used in conjunction with the plot_results
    fuction in the DCP module.

    Parameters
    ----------
    axes:  Matplotlib axis object | these axes represent

    """


def create_bcount(
    did,
    bcounts,
    depths
):
    """
    Create a file containing blow count vs depth.

    Parameters
    ----------
    did:  str | DCP ID
    bcounts:  list | list of blow counts
    depths:  list | list of depth (mm)

    """
    bcounts_ = [bcounts, depths]

    with open(
        'bcounts/{}.p'.format(did),
        'wb'
    ) as out_file:
        pickle.dump(
            bcounts_,
            out_file
        )

    update_table(
        did,
        bcount=True
    )

def get_bcounts(
    did
):
    """
    Get the file containing the results of a DCP.

    Parameters
    ----------
    did:  str | DCP ID

    Returns
    -------
    bcounts:  list | list of the blow counts and cumulative depth in the form
      [[blow counts], [depths]]

    """
    with open(
        'bcounts/{}.p'.format(did),
        'rb'
    ) as in_file:
        bcount = pickle.load(in_file)

    return bcount
