import numpy as np
import requests
import os
from astropy.io import fits
from scipy import interpolate


class Spectrum:
    def __init__(self, wavelength, flux, variance=None):
        self.wavelength = wavelength
        self.flux = flux
        if variance is not None:
            self.variance = variance

    def vac_to_air(self):
        self.wavelength_air = air_conversion(self.wavelength)

    def vel_shift(self, velocity):
        return velocity_shift(self.wavelength, velocity)

    def interpolate_spectrum(self, new_wavelength):
        ip = interpolate.InterpolatedUnivariateSpline(self.wavelength,
                                                      self.flux,
                                                      k=3, ext='zeros')
        self.wavelength = new_wavelength
        self.flux = ip(new_wavelength)


def readspec(filename, extension):
    '''
    A function designed for easy reading of fits 1D spectra files. Filenames
    can either be input as a local fits file or as a URL to files at a URL
    link.

    Returns the wavelength and flux of the 1D spectrum.
    '''

    with fits.open(filename) as file:
        # file = fits.open(filename)
        flux = file[extension].data
        header = file[extension].header

        ctype = header['CTYPE1']
        if 'CDELT1' in header:
            cdelt = header['CDELT1']
        if 'CD1_1' in header:
            cdelt = header['CD1_1']
        crpix = header['CRPIX1']
        cinit = header['CRVAL1']
        naxis1 = header['NAXIS1']

        if ctype in ['LINEAR', 'WAVELENGTH', 'AWAV']:
            wavelength = (np.arange(naxis1) + crpix ) * cdelt + cinit

        if ctype in ['WAVE']:
            new_cinit = np.log10(cinit)
            new_cdelt = np.log10(cinit+cdelt)-new_cinit
            wavelength = np.power(10., (np.arange(naxis1)-crpix) * new_cdelt + new_cinit)


        if ctype in ['LOG-LINEAR']:
            wavelength = np.power(10., (np.arange(naxis1)) * cdelt + cinit)

    return wavelength, flux


def vac_spec(filename, extension=1):
    '''
    Opens a vacuum wavelength spectrum and masks pixels with 0 flux.  Returns
    the wavelength and flux as a numpy-like array.
    '''

    wave, data = readspec(filename, extension)

    data_m = np.ma.masked_where(data == 0, data)

    return wave, data_m


def air_spec(filename, extension=1):
    '''
    Opens a vacuum wavelength spectrum, converts its wavelengths to air
    following Shetrone et al. (2015) and masks pixels with 0 flux.  Returns the
    wavelength and flux as a numpy-like array.
    '''

    wave, data = readspec(filename, extension)
    wave_air = air_conversion(wave)
    data_m = np.ma.masked_where(data == 0, data)

    return wave_air, data_m


def air_conversion(wave):
    '''
    Converts a vacuum wavelength spectrum to air following Shetrone et al.
    (2015).
    '''

    a = 0.0
    b1 = 5.792105e-2
    b2 = 1.67917e-3
    c1 = 238.0185
    c2 = 57.362

    wave = wave / 10000.

    air_conv = a + (b1 / (c1 - 1 / (wave**2.))) + \
        (b2 / (c2 - 1 / (wave**2.))) + 1

    wave_air = wave / air_conv
    wave_air = wave_air * 10000.

    return wave_air


def velocity_shift(wavelength, velocity):
    c = 2.99792458e5
    return wavelength * (1 + (velocity / c))
