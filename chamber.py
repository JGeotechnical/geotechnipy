import os
import pandas as pd
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy import and_, between

class Chamber():
    """Instantiate the Calibation Chamber class."""

    def __init__(
        self,
        hid=None,
        hdiameter=None,
        hheight=None
    ):
        """
        Instatiate the Calibration Chamber class.

        Parameters
        ----------
        hid:  str | calibration chamber ID
        hdiameter:  float | chamber diameter (cm)
        hheight:  float | chamber height (cm)

        """
        # Get the connection and soils table objects.
        connection, htable = get_table()

        # Filter the Soils Database.
        stmt = select([htable])

        ## Filter by chamber ID.
        if hid:
            stmt = stmt.where(htable.columns.hid.in_((hid)))

        ## Filter by chamber diameter.
        if hdiameter:
            stmt = stmt.where(
                between(
                    htable.columns.hdiameter,
                    hdiameter[0],
                    hdiameter[1]
                )
            )

        ## Filter by chamber height.
        if hheight:
            stmt = stmt.where(
                between(
                    htable.columns.hheight,
                    hheight[0],
                    hheight[1]
                )
            )

        ## Make the dataframe.
        self.__results = pd.read_sql(
            stmt,
            connection
        ).set_index('hid')

    @property
    def results(self):
        return self.__results

    @property
    def hdiameter(self):
        return self.__results.loc['hdiameter']

    @property
    def hheight(self):
        return self.__results.loc['hheight']


def create_chamber(
    hname,
    hrname,
    hrdate,
    hdiameter=None,
    hheight=None
):
    """
    Create a record in the Calibration Chamber CPT database.

    Parameters
    ----------
    hid:  str | calibration chamber ID
    hname:  str | calibration chamber name
    hrname:  str | name of the reference where the test came
    hrdate:  float | date the reference was publishe
    hdiameter:  float | chamber diameter (cm)
    hheight:  float | chamber height (cm)

    """
    # Remove special characters, spaces, etc. in sname and srname.
    part1 = ''.join(char for char in hname if char.isalnum()).lower()[:3]
    part2 = ''.join(char for char in hrname if char.isalnum()).lower()[:3]

    # Make the soil ID.
    hid = part1 + '-' + part2 + '-' + str(hrdate)[2:]

    # Make the engine to connect to the database.
    engine = create_engine('sqlite:///db/geotechnipy.sqlite')

    # Make the connection object.
    connection = engine.connect()

    # Make the metadata object.
    metadata = MetaData()

    # Reflect the table.
    htable = Table(
        'chamber',
        metadata,
        autoload=True,
        autoload_with=engine
    )

    # Make a record in the table.
    stmt = insert(htable).values(
        hid=hid,
        hname=hname,
        hrname=hrname,
        hrdate=hrdate,
        hdiameter=hdiameter,
        hheight=hheight
    )

    connection.execute(stmt)


def get_table():
    """
    Get the Soils database.

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

    # Make the metadata object.
    metadata = MetaData()

    # Reflect the table.
    htable = Table(
        'chamber',
        metadata,
        autoload=True,
        autoload_with=engine
    )

    return connection, htable


def update_table(
    hid,
    **kwargs
):
    """
    Updates 1 calibration chamber record.

    Parameters
    ----------
    hid:  str | calibration chamber ID

    Other Parameters
    ----------------
    hname:  str | calibration chamber name
    hrname:  str | name of the reference where the test came
    hrdate:  float | date the reference was publishe
    hdiameter:  float | chamber diameter (cm)
    hheight:  float | chamber height (cm)

    """
    # Get the connection and soils table objects.
    connection, htable = get_table()

    # Create the query statement.
    stmt = update(htable)
    stmt = stmt.where(htable.columns.hid==hid)

    # Update the database.
    connection.execute(stmt, [kwargs])
