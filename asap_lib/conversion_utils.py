import astropy.units as u
import astropy.constants
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

"""
Perform conversions 


"""


# -----------------------------------------------------------------------------------------------------------------------
def rvCoordinateCorrections(rv, telescope=None, long=None, lat=None, elevation=None, isot_time=None, year=None,
                            month=None, day=None, hour=None, minute=None, second=None, objLoc_CDS=None, objLoc_ra=None,
                            objLoc_dec=None, kind='heliocentric'):
    """
    Apply a heliocentric or barycentric correction to a radial velocity

    :param rv: Radial velocity of the object. Must be in km / s!

    :param telescope: Name of the telescope where the observations where taken. ie 'gemini_south'
    :param long: If name of the telescope is not given, then specify the longitude of the observations in degrees
    :param lat: If name of the telescope is not given, then specify the longitude of the observations in degrees
    :param elevation: If name of the telescope is not given, then specify the elevation of the observations in meters

    :param isot_time: UTC time of the observation in ISO extended format. ie 'year:month:dayThour:minute:second'
    :param year: If isot_time is not given, specify the year of the observation in UTC
    :param month: If isot_time is not given, specify the month of the observation in UTC
    :param day: If isot_time is not given, specify the day of the observation in UTC
    :param hour: If isot_time is not given, specify the hour of the observation in UTC
    :param minute: If isot_time is not given, specify the minute of the observation in UTC
    :param second: If isot_time is not given, specify the second of the observation in UTC

    :param objLoc_CDS: Location of the object on the sky (ra, dec, J2000) in CDS_objLoc "hr min sec degree min sec"
     format as if you copy the location directly from CDS
    :param objLoc_ra: If objLoc_CDS is not given, ra of the object in degrees
    :param objLoc_dec: If objLoc_CDS is not given, dec of the object in degrees

    :param kind: Type of correction. 'heliocentric' or 'barycentric'

    :return: Corrected radial velocity in km /s
    """

    # ----------------- Observation Location
    if telescope == None:
        if long == None or lat == None or elevation == None:
            print('Must provide observation location. That is, telescope name or longtime, latitude and elevation')
            return

    if telescope != None and telescope not in EarthLocation.get_site_names():
        print('Telescope not in known sites. Please manually enter location information (longitude, latitude '
              'and elevation)')
        return

    if telescope != None:
        obs_location = EarthLocation.of_site(telescope)
    else:
        obs_location = EarthLocation.from_geodetic(lat=lat, lon=long, height=elevation)

    # ----------------- Observation Time
    if isot_time == None:
        if year == None or month == None or day == None or hour == None or minute == None or second == None:
            print('Must provide detailed UTC observation time (year, month, day, hour, minute, second)')
            return

    if isot_time != None:
        obs_time = Time(isot_time, scale='utc')
    else:
        time = year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':' + second
        obs_time = Time(time, scale='utc')

    # ----------------- Observation Location on Sky
    if objLoc_CDS == None:
        if objLoc_ra == None or objLoc_dec == None:
            print('Must provide detailed object location on the sky')
            return

    if objLoc_CDS != None:
        raH = objLoc_CDS.split()[0]
        raM = objLoc_CDS.split()[1]
        raS = objLoc_CDS.split()[2]

        decD = objLoc_CDS.split()[3]
        decM = objLoc_CDS.split()[4]
        decS = objLoc_CDS.split()[5]

        sc = SkyCoord(ra=raH+'h'+raM+'m'+raS+'s', dec=decD+'d'+decM+'m'+decS+'s')
    else:
        sc = SkyCoord(ra=objLoc_ra, dec=objLoc_dec)

    # ----------------- Type of correction
    if kind not in ['heliocentric', 'barycentric']:
        print('Correction type must be either \'heliocentric\' or \'barycentric\' ')
        return

    # ----------------- Get the correction
    vcorr = sc.radial_velocity_correction(kind=kind, obstime=obs_time, location=obs_location)

    # ----------------- Apply the correction
    if kind == 'barycentric':
        corrected_rv = (rv*u.km/u.s) + vcorr + ((rv*u.km/u.s)*vcorr) / astropy.constants.c
        return corrected_rv.to(u.km / u.s).value

    else:
        corrected_rv = (rv*u.km/u.s) + vcorr
        return corrected_rv.to(u.km / u.s).value
