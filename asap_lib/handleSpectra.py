import os
import pickle
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table

"""
Handle Spectra

- If you want to read, write, or convert a spectrum to another format then you've come to the right place
"""


# -----------------------------------------------------------------------------------------------------------------------
def read_spec(spath, ftype=None, IFU=0):
    """
     A function to read a GRACES spectrum and extract the wavelengths, flux, and error

    -- If the wavelengths are in nm convert them to angstroms

    Supported data structures/ file types: 'fits', 'cfits', 'ghostfits', 'bin', 'xy', 'xyz'

    Note: The fits files from GHOST may have different extensions.
    If you are extracting data from GHOST fits files be aware of the IFU mode (defaults to 0)

    'xyz' is a new file type being introduced which has wavelength, flux, and error data in three columns of that order.
    It is identical to xy files with the addition of a third column for error.
    This is meant to resolve confusion with 'xy' files which have different numbers of columns
    """

    # -------- Automatic Detect File Type
    if ftype is None:
        ftype = spath.split('.')[-1]
        print('Automatically detecting file type.... I think it is .{}'.format(ftype))
    # -----------------
    if ftype == 'fits':
        # Read in a .fits spectrum and pull the wavelength, flux, and error arrays
        spectra = fits.open(spath)
        wave = spectra[0].data[0]
        flux = spectra[0].data[1]
        err = spectra[0].data[4]

        # Convert from nm to A if needed
        if np.max(wave) < 3000.:
            wave *= 10.

        return wave, flux, err

    # -----------------
    if ftype == 'cfits':
        spectra = fits.open(spath)
        hdr = spectra[0].header
        naxis1 = hdr['NAXIS1']
        crval1 = hdr['CRVAL1']
        cdelt1 = hdr['CDELT1']

        # Create a wavelength array and store information about it
        wave = np.linspace(crval1, crval1 + (naxis1 * cdelt1), naxis1)
        flux = spectra[0].data

        # Convert from nm to A if needed
        if np.max(wave) < 3000.:
            wave = 10. * wave

        return wave, flux

    # -----------------
    if ftype == 'ghostfits':
        spectra = fits.open(spath)

        wave = spectra[7].data
        flux = spectra[5].data[:, IFU]
        err = spectra[6].data[:, IFU]

        # Convert from nm to A if needed
        if np.max(wave) < 3000.:
            wave *= 10.

        return wave, flux, err

    # -----------------
    if ftype == 'bin':
        spectra = pickle.load(open(spath, 'rb'))
        wave = spectra['wave']
        flux = spectra['flux']
        err = spectra['err']
        # Convert from nm to A if needed
        if np.max(wave) < 3000.:
            wave *= 10.

        return wave, flux, err

    # -----------------
    if ftype == 'xy':

        with open(spath, 'r') as f:
            lines = f.readlines()

        wave = []
        flux = []
        for i, l in enumerate(lines):
            if '#' not in l:
                split = l.split()
                wave.append(float(split[0]))
                flux.append(float(split[1]))

        wave, flux = np.asarray(wave), np.asarray(flux)

        # Convert from nm to A if needed
        if np.max(wave) < 3000.:
            wave = 10. * wave

        return wave, flux

    # -----------------
    if ftype == 'xyz':

        with open(spath, 'r') as f:
            lines = f.readlines()

        wave = []
        flux = []
        err = []
        for i, l in enumerate(lines):
            if '#' not in l:
                split = l.split()
                wave.append(float(split[0]))
                flux.append(float(split[1]))
                err.append(float(split[2]))

        wave, flux, err = np.asarray(wave), np.asarray(flux), np.asarray(err)

        # Convert from nm to A if needed
        if np.max(wave) < 3000.:
            wave = 10. * wave

        return wave, flux, err


# -----------------------------------------------------------------------------------------------------------------------
def i2xy(spectra, spath, xypath):
    """
    Convert OPERA processed GRACES i.fits files to a .xy file since MOOG
    will ultimately be used.

    Parameters:
        spectra: array or list
            File names of the spectra. Ex: ['NYYYYMMD1i.fits', 'NYYYYD2i.fits']
        spath: str
            Path to the i.fits spectra
        xypath: str
            Path to the output .xy spectra
    Returns: .xy files in xypath
    """
    if type(spectra) == str:
        spectra = [spectra]

    # -----------------  Examine each spectrum
    for s in spectra:
        wave, flux, err = read_spec(os.path.join(spath, s))

        # Save spectrum as .xy file
        colheads = ['wave', 'flux', 'err']
        table = Table([wave, flux, err], names=colheads)
        # pickle.dump(table, open(xypath + s[:-6] + '.xy','wb'))
        ascii.write(table, xypath + s[:-6] + '.xy')


# -----------------------------------------------------------------------------------------------------------------------
def bin2xy(spectra, spaths, xypaths, xytype=None):
    """
    Convert binary spectrum files to a .xy file since MOOG
    will ultimately be used.

    Parameters:
        spectra: array or list
            File names of the spectra. Ex: ['PXXX.bin', 'PYYY.bin']
        spaths: str
            Path to the binary spectra
        xypaths: str
            Path to the output .xy spectra
        xytype: str
            Format specifier of the output .xy spectra
    Returns: .xy files in xypath
    """
    if type(spectra) == str:
        spectra = [spectra]

    # -----------------  Examine each spectrum
    for i, s in enumerate(spectra):
        wave, flux, err = read_spec(os.path.join(spaths[i], s), ftype='bin')

        # name = s.split('.')[0]
        name = s.split('.comb')[0]

        wave = np.asarray(wave)
        flux = np.asarray(flux)

        # Save spectrum as .xy file
        if not xytype:
            colheads = ['wave', 'flux']
            table = Table([wave, flux], names=colheads)
            # pickle.dump(table, open(xypath + s[:-6] + '.xy','wb'))
            ascii.write(table, os.path.join(xypaths[i], name + '.xy'))

        if xytype == 'MOOG':
            lines = []
            for j in range(len(wave)):
                if j != len(wave) - 1:
                    line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + '\n'
                    lines.append(line)
                else:
                    line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8))
                    lines.append(line)

            with open(os.path.join(xypaths[i], name + '.xy'), 'w') as file:
                file.writelines(lines)


# -----------------------------------------------------------------------------------------------------------------------
def bin2xy_v2(spec, specpath, savepath, xytype=None):
    """
    Convert binary spectrum files to a .xy file since MOOG
    will ultimately be used.

    Parameters:
        spec: str
            File names of the spectra. Ex: 'PXXX'
        specpath: str
            Full Path to the binary spectra. Ex: '..Data/PXXX.bin'
        savepath: str
            Path to the output .xy spectra. Ex: '..Data/final_spectra/'
        xytype: str
            Format specifier of the output .xy spectra
    Returns: .xy files in xypath
    """

    wave, flux, err = read_spec(os.path.join(specpath, spec + '.bin'), ftype='bin')

    wave = np.asarray(wave)
    flux = np.asarray(flux)

    # Save spectrum as .xy file
    if not xytype:
        colheads = ['wave', 'flux']
        table = Table([wave, flux], names=colheads)
        ascii.write(table, os.path.join(savepath, spec + '.xy'))
        print('Saving .xy file to ' + os.path.join(savepath, spec + '.xy'))

    if xytype == 'MOOG':
        lines = []
        for j in range(len(wave)):
            if j != len(wave) - 1:
                line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + '\n'
                lines.append(line)
            else:
                line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8))
                lines.append(line)

        with open(os.path.join(savepath, spec + '.xy'), 'w') as file:
            file.writelines(lines)
            print('Saving .xy file to ' + os.path.join(savepath, spec + '.xy'))


# -----------------------------------------------------------------------------------------------------------------------
def fits2xy(spectra, spaths, xypaths, xytype=None):
    """
    Convert fits spectrum files to a .xy file since MOOG
    will ultimately be used.

    Parameters:
        spectra: array or list
            File names of the spectra. Ex: ['PXXX.fits', 'PYYY.fits']
        spaths: str
            Path to the fits spectra
        xypaths: str
            Path to the output .xy spectra
        xytype: str
            Format specifier of the output .xy spectra
    Returns: .xy files in xypath
    """
    if type(spectra) == str:
        spectra = [spectra]

    # ----------------- Examine each spectrum
    for i, s in enumerate(spectra):

        name = s.split('.fits')[0]

        wave, flux = read_spec(os.path.join(spaths[i], s), ftype='cfits')

        wave = np.asarray(wave)
        flux = np.asarray(flux)

        # Save spectrum as .xy file
        if not xytype:
            colheads = ['wave', 'flux']
            table = Table([wave, flux], names=colheads)
            ascii.write(table, os.path.join(xypaths[i], name + '.xy'))

        if xytype == 'MOOG':
            lines = []
            for j in range(len(wave)):
                if j != len(wave) - 1:
                    line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + '\n'
                    lines.append(line)
                else:
                    line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8))
                    lines.append(line)

            with open(os.path.join(xypaths[i], name + '.xy'), 'w') as file:
                file.writelines(lines)


# -----------------------------------------------------------------------------------------------------------------------
def bin2DAOxy(spectra, spaths, xypaths, lims=None, masks=None):
    """
    Convert bin spectrum to DAO xy spectrum

    Inputs
    -------
    spectra: array or list
            File names of the spectra. Ex: ['PXXX.fits', 'PYYY.fits']
    spaths: str
            Path to the binary spectra
    xypaths: str
            Path to the output .xy spectra
    lims: tuple (lower, upper)
            Wavelength limits on the saved spectra
    masks: list
    """
    if type(spectra) == str:
        spectra = [spectra]

    # ----------------- Examine each spectrum
    for i, s in enumerate(spectra):
        wave, flux, err = read_spec(os.path.join(spaths[i], s), ftype='bin')
        # name = s.split('.')[0]
        name = s.split('.comb')[0]

        wave = np.asarray(wave)
        flux = np.asarray(flux)

        if lims:
            good = np.where((wave >= lims[0]) & (wave <= lims[1]))[0]
            wave = wave[good]
            flux = flux[good]

        if masks:
            for m in masks:
                bad = np.where((wave >= m[0]) & (wave <= m[1]))[0]
                flux[bad] = 1.0

        lines = []
        for j in range(len(wave)):
            if j != len(wave) - 1:
                line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + '\n'
                lines.append(line)
            else:
                line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8))
                lines.append(line)

        with open(os.path.join(xypaths[i], name + '_daoM.xy'), 'w') as file:
            file.writelines(lines)


# -----------------------------------------------------------------------------------------------------------------------
def write2xy(wave, flux, spath):
    """
    Write a .xy spectral file from wavelength, and flux arrays

    :param wave:  (array-like) Wavelength array
    :param flux: (array-like) Flux array corresponding to the wavelengths in wave
    :param spath: (str) Path to file you wish to save.  Ie Data/name

    Saves a .xy file at spath
    """
    if len(wave) != len(flux):
        raise TypeError("Wavelength, flux, and error arrays must be the same length")

    lines = []
    for j in range(len(wave)):
        if j != len(wave) - 1:
            line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + '\n'
            lines.append(line)
        else:
            line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8))
            lines.append(line)

    with open(spath + '.xy', 'w') as file:
        file.write("# wavelength flux variance \n")
    with open(spath + '.xy', 'a') as file:
        file.writelines(lines)

# -----------------------------------------------------------------------------------------------------------------------
def xyz2xy(spath):
    """
    A convenience function to convert a .xyz file to a .xy file
    Writes a .xy spectral file

    :param spath: (str) Path to file you wish to save.  Ie Data/name

    Saves a .xy file at spath
    """
    wave, flux, err = read_spec(spath, ftype='xyz')
    write2xy(wave, flux, spath)


# -----------------------------------------------------------------------------------------------------------------------
def write2xyz(wave, flux, err, spath):
    """
    Write a .xyz spectral file from wavelength, flux, and flux error arrays

    :param wave:  (array-like) Wavelength array
    :param flux: (array-like) Flux array corresponding to the wavelengths in wave
    :param err: (array-like) Flux error array corresponding to the wavelengths in wave
    :param spath: (str) Path to file you wish to save.  Ie Data/name

    Saves a .xyz file at spath
    """
    if len(wave) != len(flux) or len(wave) != len(err):
        raise TypeError("Wavelength, flux, and error arrays must be the same length")

    lines = []
    for j in range(len(wave)):
        if j != len(wave) - 1:
            line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + ' ' + str(round(err[j], 8)) + '\n'
            lines.append(line)
        else:
            line = str(round(wave[j], 8)) + ' ' + str(round(flux[j], 8)) + ' ' + str(round(err[j], 8))
            lines.append(line)

    with open(spath + '.xyz', 'w') as file:
        file.write("# wavelength flux variance \n")
    with open(spath + '.xyz', 'a') as file:
        file.writelines(lines)
    #with open(spath + '.xyz', 'w') as file:
        #file.writelines(lines)


# -----------------------------------------------------------------------------------------------------------------------
def write2bin(wave, flux, err, spath):
    """
    Write a .bin spectral file from wavelength, flux, and flux error arrays

    :param wave:  (array-like) Wavelength array
    :param flux: (array-like) Flux array corresponding to the wavelengths in wave
    :param err: (array-like) Flux error array corresponding to the wavelengths in wave
    :param spath: (str) Path to file you wish to save.  Ie Data/name

    Saves a .bin file at spath
    """
    if len(wave) != len(flux) or len(wave) != len(err):
        raise TypeError("Wavelength, flux, and error arrays must be the same length")

    colheads = ['wave', 'flux', 'err']
    table = Table([wave, flux, err], names=colheads)

    pickle.dump(table, open(spath + '.bin', 'wb'))


# -----------------------------------------------------------------------------------------------------------------------
def xyz_2_bin(spectrum, spath):
    """
    Convert a .xyz file with columns wave, flux, err to a .bin file
    -----

    :param spectrum: (str) Path to the spectrum
    :param spath: (str) Name of the spectrum (excluding file extension)

    saves a file at spath/spectrum.bin
    """
    # ---- Open xyz file
    with open(spath + spectrum + '.xyz', 'r') as f:
        lines = f.readlines()

        wave = []
        flux = []
        err = []
        for i, l in enumerate(lines):
            if '#' not in l:
                split = l.split()
                wave.append(float(split[0]))
                flux.append(float(split[1]))
                err.append(float(split[2]))

        wave, flux, err = np.asarray(wave), np.asarray(flux), np.asarray(err)

    # ---- Write to table
    colheads = ['wave', 'flux', 'err']
    table = Table([wave, flux, err], names=colheads)

    pickle.dump(table, open(spath + spectrum + '.bin', 'wb'))
