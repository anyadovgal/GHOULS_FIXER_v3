import os
import sys
# from tool_uri import uri_match
from scipy.interpolate import interp1d
import scipy.signal as spsi
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy import interpolate
import scipy.stats as spst
import copy
from scipy.ndimage import median_filter, gaussian_filter1d, gaussian_filter

# ----------------- Import the other files of functions
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# from asap_lib.constants import *
# from asap_lib.handleSpectra import *
# from asap_lib.radial_velocity import *

from asap_lib.constants import *
from asap_lib.handleSpectra import *
from asap_lib.radial_velocity import *



# -----------------------------------------------------------------------------------------------------------------------
def extract(name, identifier, params):
    # ----------------- Extract values from a list of parameters for a particular star
    if (type(name) == str) or (type(name) == float):
        name = [name]
    n = len(params)

    lists = [[] for _ in range(n)]
    for i in range(len(name)):
        good = np.where(name[i] == np.array(identifier))[0]

        for j, l in enumerate(lists):
            l.append(np.array(params[j])[good])

    return lists


# -----------------------------------------------------------------------------------------------------------------------
def wavg(vals, weights, rtN=False):
    """
    Compute a weighted average w/ error
    """
    var = 1. / (weights ** 2)
    wx = vals * var
    w_avg = wx.sum(axis=0) / var.sum(axis=0)
    w_err = 1. / np.sqrt(var.sum(axis=0))
    std = vals.std(axis=0)

    if rtN:
        try:
            nvals = len(vals)
            err = np.sqrt(w_err ** 2 + std ** 2) / np.sqrt(nvals)
        except:
            err = np.sqrt(w_err ** 2 + std ** 2)
    else:
        err = np.sqrt(w_err ** 2 + std ** 2)

    return w_avg, err


# -----------------------------------------------------------------------------------------------------------------------
def find_nearest(arr, value):
    """
    In the given wavelength array, find the index and value of the
    wavelength that most closely matches "value"
    (most closely matches = the smallest difference between array wavelengths and
    "value")

    Returns
    - idx: index of array where the wavelength is closest to "value"
    - array[idx]: value of the wavelength that is closest to "value"
    """
    array = np.copy(arr)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return [idx, array[idx]]


# -----------------------------------------------------------------------------------------------------------------------
def spec_ex(spectra,
            spath,
            s_info=False,
            s_plot=False,
            xlims=None,
            ylims=None,
            smo=None):
    """
    Examine some spectra
    ----------

    spectra: Input spectra you want to see. Accept lists or strings
    for star name only. Two column Wavelength/Flux or binary formats only
    spath: Path to spectra
    s_info: True/False. If true, print/save basic info about spectra
    s_plot: True/False. Plot spectra if desired
    smo: gaussian smoothing parameter
    xlims: List. Plot x limits (Ex: [500,1000])
    ylims: List. Plot y limits (Ex: [0,1])

    """
    if type(spectra) == str:
        spectra = [spectra]

    s_params = []

    # ----------------- Examine each spectrum
    for s in spectra:

        # ----------------- Read in the data
        wave, flux, err = read_spec(os.path.join(spath, s), ftype='bin')

        # ----------------- Start plot stuff
        if s_plot is True:
            plt.figure(figsize=(8, 5))

            # Don't plot more data than you need to
            if xlims is not None:
                good = np.where(np.logical_and(wave >= xlims[0],
                                               wave <= xlims[1]))[0]
                wave = wave[good]
                flux = flux[good]
                # err = err[good]

                plt.xlim(xlims[0], xlims[1])
            if ylims is not None:
                plt.ylim(ylims[0], ylims[1])

            # Apply a gaussian smoothing filter if desired
            if smo is not None:
                wave = gaussian_filter(wave, sigma=smo)
                flux = gaussian_filter(flux, sigma=smo)

            # Plot the data
            plt.plot(wave, flux, color='k', linewidth=0.5, linestyle='-')

            # Label and clean up the plot if needed
            #             plt.text(np.min(wave)+50,1.1,s)
            plt.title('s_plot=True\n' + s)
            plt.xlabel('Wavelength [$\AA$] ')
            plt.ylabel('Flux')

        # ----------------- Spectra information
        if s_info is True:
            s_param = [['Name', s]]
            min_w = np.min(wave)
            s_param.append(['Min (A)', round(min_w, 4)])
            max_w = np.max(wave)
            s_param.append(['Max (A)', round(max_w, 4)])
            len_w = len(wave)
            s_param.append(['N points', len_w])
            sep = []
            for i in range(len(wave) - 1):
                sep.append(wave[i + 1] - wave[i])
            # avg_sep = (max_w - min_w)/len_w
            avg_sep = float(np.mean(sep))
            s_param.append(['Mean Separation (A)', round(avg_sep, ndigits=6)])

            s_params.append(s_param)

    # ----------------- Save Information as text file
    if s_info is True:
        s_params = np.array(s_params, dtype=object)
        return s_params


# -----------------------------------------------------------------------------------------------------------------------
def spec_trim(spectra,
              spath,
              xlims,
              tpath=None,
              new_wave=False,
              save=False):
    """
    Trim spectra / a spectrum to a certain wavelength range
    ------------

    spectra: Input spectra you want to see. Accepts list or string
    for star name only. Two column Wavelength/Flux or binary formats only
    spath: Path to spectra
    xlims: List. Trim x (wavelength) limits (Ex: [500,1000])
    tpath: Path to save trimmed spectra into
    out: True/False. Output the new spectrum? Careful with loops here...better to pass individual spectra
    save: True/False. Actually save the new files or just do all this for no reason

    """

    if type(spectra) == str:
        spectra = [spectra]

    spec_lengths = []

    # ----------------- For each spectrum
    for s in spectra:

        # ----------------- Read in the data
        data = pickle.load(open(spath + s, 'rb'))

        # ----------------- Get data (wavelenghts, fluxes, and errors)
        wave = data['wave']
        flux = data['flux']
        err = data['err']

        # ----------------- Trim spectrum (select data within the wavelength range)
        good = np.where((wave >= xlims[0]) & (wave <= xlims[1]))
        s_wave = wave[good]
        s_flux = flux[good]
        s_err = err[good]
        data = [s_wave, s_flux, s_err]
        spec_lengths.append(len(s_wave))

        # ----------------- Save new trimmed spectrum
        if save is True:
            colheads = ['wave', 'flux', 'err']
            table = Table(data, names=colheads)

            print('Saving trimmed spectrum to ' + tpath + s[:-4] + '.trim.bin')
            pickle.dump(table, open(tpath + s[:-4] + '.trim.bin', 'wb'))

    # ----------------- End of the spectrum loop

    if new_wave is True:  # Output the new wavelength grid for future interpolation
        # Choose the highest sampled spectrum (longest spectrum) to define the length of the new wavelength array
        new_length = np.max(spec_lengths)

        # Create the new wavelength array
        new_wave = np.linspace(xlims[0], xlims[1], new_length)

        return new_wave


# -----------------------------------------------------------------------------------------------------------------------
def spec_interp(spectra,
                spath,
                i_wave=None,
                ipath=None,
                save=False):
    """
    Interpolate spectra to a new wavelength grid
    --------

    spectra: Input spectra you want to see. Accepts list or string
    for star name only. Two column Wavelength/Flux or binary formats only
    spath: Path to spectra
    i_wave: Wavelength array to interpolate onto
    ipath: Path to save interpolated spectra into
    save: True/False. Actually save the new files or just do all this for no reason

    """

    if type(spectra) == str:
        spectra = [spectra]

    # -----------------Examine each spectrum
    for s in spectra:

        # ----------------- Read in the data
        wave, flux, err = read_spec(os.path.join(spath, s), ftype='bin')

        # ----------------- Interpolate data onto new wavelength grid
        i_flux = np.interp(i_wave, wave, flux)
        i_err = np.interp(i_wave, wave, err)

        data = [i_wave, i_flux, i_err]

        # ----------------- Save new spectrum
        if save is True:
            colheads = ['wave', 'flux', 'err']
            table = Table(data, names=colheads)

            if ipath is None:
                ipath = spath

            print('Saving interpolated spectrum to ' + ipath + s[:-4] + '.i.bin')
            pickle.dump(table, open(ipath + s[:-4] + '.i.bin', 'wb'))


# -----------------------------------------------------------------------------------------------------------------------
def interp_2_constant_dwgrid(wave, flux, err, kind='cubic', saveName=None):
    """
    Interpolate a spectrum onto a wavelength grid with constant spacing in wavelength (dw)
    ----------
    Example use: before radial velocity correcting a spectrum the spectrum must be on a wavelength grid with constant
    spacing in wavelength (dw)

    :param list wave: Wavelength array
    :param list flux: Flux array corresponding to the wavelength array
    :param list err:  Error array corresponding to the wavelength array
    :param str kind: Kind of the interpolation. ie 'linear' 'quadratic' 'cubic'
    :param str saveName: Name and path to save the interpolated spectrum to

    :return: If saveName = None, the  interpolated arrays will be returned

    """
    # ----------------- Interpolate the flux and error spectra
    interpFunc_flux = interpolate.interp1d(wave, flux, kind=kind)
    interpFunc_err = interpolate.interp1d(wave, err, kind=kind)

    # ----------------- Define a new wavelength grid
    new_w = np.linspace(min(wave), max(wave), len(wave))

    # ----------------- Evaluate the flux and error functions along the new wavelength grid
    new_f = interpFunc_flux(new_w)
    new_e = interpFunc_err(new_w)

    # ----------------- Save or return
    if saveName:
        write2bin(new_w, new_f, new_e, saveName)

    else:
        return new_w, new_f, new_e


# -----------------------------------------------------------------------------------------------------------------------
def list_duplicates_of(seq, item):
    """
    Search a list of objects (seq) and return the indices corresponding to all instances of (item)
    """
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)

        except ValueError:
            break

        else:
            locs.append(loc)
            start_at = loc

    return locs


# -----------------------------------------------------------------------------------------------------------------------
def spec_stack(spectra, name, occurs=None, spath=None, combpath=None, print_prog=False):
    """
    Stack (combine) the visit spectra using a weighted average
    --------
    Spectra must be interpolated to the same wavelength grid

    spectra: list of all spectra files
    name: object name
    occurs: files corresponding to the visits of name (required if there are multiple objects)
    spath: path to the spectral files to combine
    combpath: path to the stacked spectra
    """

    wave = []
    flux = []
    err = []

    # -----------------
    # If there are multiple objects:
    if occurs is not None:

        # For each visit:
        for i in range(0, len(occurs)):

            if print_prog is True:
                print('Object: ', name, ' - Handling file ', spectra[occurs[i]], ' (', i + 1, 'of', len(occurs), ')')

            # Load data from that visit
            data = pickle.load(open(spath + spectra[occurs[i]], 'rb'))

            if i == 0:
                wave = data['wave']

            flux.append(data['flux'])
            err.append(data['err'])

    # -----------------
    # If there is just one object:
    else:
        # For each visit:
        for i, s in enumerate(spectra):

            if print_prog is True:
                print('Handling file ', s, ' (', i + 1, 'of', len(spectra), ')')

            data = pickle.load(open(spath + s, 'rb'))

            # Load data from that visit
            if i == 0:
                wave = data['wave']

            flux.append(data['flux'])
            err.append(data['err'])

            # -----------------
    wave = np.array(wave)
    err = np.array(err)

    # Get the weighted average error of each visit at each wavelength
    avg_err = np.sqrt(1. / np.sum(1. / err ** 2., axis=0))

    # Get the weighted average of the flux from each visit at each wavelength
    flux = np.array(flux)
    avg_flux = np.average(flux, axis=0, weights=1. / err ** 2.)

    # ----------------- Save new spectrum
    data = [wave, avg_flux, avg_err]
    colheads = ['wave', 'flux', 'err']

    table = Table(data, names=colheads)

    if combpath is not None:
        print('Saving combined spectrum to ' + combpath + name + '.comb.bin')
        pickle.dump(table, open(combpath + name + '.comb.bin', 'wb'))


# # -----------------------------------------------------------------------------------------------------------------------
# def spec_chop(wave, flux, elems_list):
#     """
#     Chop a spectrum into sections around specific spectral lines
#
#     wave: the wavelength array of the spectrum to chop
#     flux: the corresponding flux array of the spectrum to chop
#     elems: a list of elements around whose spectral lines we want to chop
#     (must be some combination of ['Mg', 'CaT', 'Ha', 'Hb', 'Hepsilon', Hgamma', 'Hdelta', 'SrII_4215.524',
#     'BaII_4554.030', 'FeII_4923.922', 'FeI_4957.596', 'FeII_5018.435', 'FeI_5269.537', 'CaI_6122.217' ] )
#     """
#
#     waves = []
#     fluxes = []
#
#     for e in elems_list:
#
#         # ----------------- Magnesium b lines: 5167A, 5172A, 5183A
#         if e == 'Mg':
#             good = np.where((wave >= 5100.) & (wave <= 5250.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Sodium D lines: 5896A, 5890A
#         if e == 'Na':
#             good = np.where((wave > 5850.) & (wave < 5940.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Calcium Triplet: 8498A, 8542A and 8662A
#         if e == 'CaT':
#             good = np.where((wave >= 8450.) & (wave <= 8700.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- H-alpha: 6565A
#         if e == 'Ha':
#             good = np.where((wave >= 6520.) & (wave <= 6600.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#             # ----------------- H-beta: 4863A
#         if e == 'Hb':
#             good = np.where((wave >= 4820.) & (wave <= 4900.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#             # ----------------- Magnesium (telluric)
#         if e == 'Mg_Tell':
#             good = np.where((wave >= 4420.) & (wave <= 4500.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Oxygen (telluric)
#         if e == 'O_Tell':
#             good = np.where((wave >= 7750.) & (wave <= 7800.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- H-epsilon: 3970.072
#         if e == 'Hepsilon':
#             good = np.where((wave >= 3925) & (wave <= 4015.))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- H-delta: 4101.734
#         if e == 'Hdelta':
#             good = np.where((wave >= 4056) & (wave <= 4146))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Sr-II: 4215.524
#         if e == 'SrII_4215.524':
#             good = np.where((wave >= 4080) & (wave <= 4170))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- H-gamma: 4340.462
#         if e == 'Hgamma':
#             good = np.where((wave >= 4295) & (wave <= 4385))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Ba-II: 4554.030
#         if e == 'BaII_4554.030':
#             good = np.where((wave >= 4509) & (wave <= 4599))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Fe-II: 4923.922
#         if e == 'FeII_4923.922':
#             good = np.where((wave >= 4879) & (wave <= 4969))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Fe-I: 4957.596
#         if e == 'FeI_4957.596':
#             good = np.where((wave >= 4912) & (wave <= 5002))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Fe-II: 5018.435
#         if e == 'FeII_5018.435':
#             good = np.where((wave >= 4973) & (wave <= 5063))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Fe-I: 5269.537
#         if e == 'FeI_5269.537':
#             good = np.where((wave >= 5224) & (wave <= 5314))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#         # ----------------- Ca-I: 6122.217
#         if e == 'CaI_6122.217':
#             good = np.where((wave >= 6077) & (wave <= 6167))[0]
#             waves.append(wave[good])
#             fluxes.append(flux[good])
#
#     return waves, fluxes
#

# -----------------------------------------------------------------------------------------------------------------------
def deg_res(spectra, spath, start_res, new_res, resample=False, save_smo_spec=True, out_path=None):
    if type(spectra) == str:
        spectra = [spectra]

    if out_path is None:
        out_path = spath

    for s in spectra:

        wave, flux, err = read_spec(os.path.join(spath, s), ftype='bin')

        x0 = wave[0]  # Start wavelength
        xN = wave[-1]  # End wavelength
        xc = (xN + x0) / 2.  # Central wavelength

        xpix = (xN - x0) / float(len(wave))  # Angstroms per pixel

        # Approximate the size of our resolution element delta lambda
        dl = xc / start_res

        # Convolve with Gaussian with FWHM=N*dl (in pixels) where N = Original Res/New Res
        conv = start_res / new_res

        dl_l = dl * conv

        sig = (dl_l / xpix) / (2. * (2. * np.log(2.) ** 0.5))  # sigma corresponding to FWHM

        if resample:
            # Low resolution flux on a resolution appropriate wavelength scale
            sampling = 2.5  # nyquist sampling
            new_dl = dl_l / sampling
            lr_w = np.linspace(x0, xN, (xN - x0) / new_dl)
            f = interp1d(wave, flux, bounds_error=False, fill_value=0.)
            lr_f = f(lr_w)
            lr_e = f(lr_w)
            wave = np.copy(lr_w)
        else:
            # Low resolution flux on original wavelength scale/sampling
            lr_f = gaussian_filter1d(flux, sig, axis=-1, order=0, mode='constant')
            lr_e = gaussian_filter1d(err, sig, axis=-1, order=0, mode='constant')

        if save_smo_spec is True:
            # Assuming the spectrum file ends in .bin:
            spec_name = s.split('.')[0]
            if resample:
                name = spec_name + '_%s' % int(new_res) + '_rs.bin'

            else:
                name = spec_name + '_%s' % int(new_res) + '.bin'
            colheads = ['wave', 'flux', 'err']
            table = Table([wave, lr_f, lr_e], names=colheads)
            pickle.dump(table, open(out_path + name, 'wb'))


# # -----------------------------------------------------------------------------------------------------------------------
# def pyfxcor(wave, flux, s_wave, s_flux, v_tol=5.0, print_vel=False, plot_shift=False, return_corr_wave=False):
#     """
#     Find the radial velocity shift between the observed and synthetic spectra.
#     Restrict the observed and synthetic spectra to regions around certain spectral features
#
#     Inputs
#     ------
#     waves: An array of wavelengths for the observed spectrum
#     fluxes: The fluxes corresponding to the wavelengths in wave
#
#     s_waves: An array of wavelengths for the synthetic / reference spectrum
#     s_fluxes: The fluxes corresponding to the wavelengths in s_waves
#
#     plot_shift: If True plot the shifted spectrum for each window (chopped section of spectrum)
#     """
#
#     # ----------------- Make copies of the arrays to not overwrite things
#     t_w = np.copy(wave)  # Temporary observed wavelength array
#     t_f = np.copy(flux)  # Temporary observed flux arrays
#     t_s_w = np.copy(s_wave)  # Temporary synthetic wavelength array
#     t_s_f = np.copy(s_flux)  # Temporary synthetic  flux arrays
#
#     # ----------------- Interpolate the synthetic spectra on to the observed spectra grid
#     '''
#     For the synthetic spectrum find a function (f) which gives synthetic flux as a function of synthetic wavelength
#     '''
#     f = interpolate.interp1d(t_s_w, t_s_f, kind='cubic', bounds_error=False, fill_value=1.0)
#     '''Evaluate this function at the wavelengths of the observed wavelength array. This generates an array of
#     synthetic fluxes corresponding to the same wavelength grid as the observed fluxes (new_tsf) '''
#     new_tsf = f(t_w)
#
#     # -----------------
#     obs = np.copy(t_f)
#     synth = np.copy(new_tsf)
#
#     # ----------------- Regularize the datasets by subtracting off the mean and dividing by the standard deviation
#     obs = obs - obs.mean()
#     obs = obs / obs.std()
#
#     synth = synth - synth.mean()
#     synth = synth / synth.std()
#
#     # ----------------- Find the cross-correlation in pixel values
#     '''The cross-correlation works by shifting one array relative to the other and calculating the correlation (
#     similarity) between the shifted arrays
#
#     First, find the cross-correlation between the  observed spectrum and the synthetic spectrum.  xcorr is an array
#     of correlation values, each corresponding to a different pixel shift between the spectra '''
#     xcorr = spsi.correlate(obs, synth, method='fft')
#
#     # Create an array (dp) which contains the pixel shifts corresponding to each correlation value in xcorr
#     nsamples = obs.size
#     dp = np.arange(1 - nsamples, nsamples)
#
#     '''We want the pixel shift corresponding to the highest correlation value - xcorr.argmax() gives the index of the
#     xcorr array corresponding to the highest correlation value - dp[xcorr.argmax()] gives the pixel shift
#     corresponding to the highest correlation value.  We want to select the inverse of this shift to correct for the
#     radial velocity. '''
#     pix_shift = -dp[xcorr.argmax()]
#
#     # ----------------- Convert the pixel shift to a velocity
#     # Calculate the conversion between pixels and velocity
#
#     # Find the average change in wavelength per pixel
#     dispersions = []
#     for i in range(len(t_w) - 1):
#         dispersions.append(t_w[i + 1] - t_w[i])
#     dispersion = np.mean(dispersions)
#
#     # Find the change in wavelength corresponding to the pixel shift
#     d_lam = dispersion * pix_shift
#
#     # Let the rest wavelength be the mean of the synthetic wavelength array
#     lam = np.median(t_s_w)
#     print(lam)
#
#     # Get the velocity correction
#     vel = s_o_l * (d_lam / lam)
#
#     if print_vel:
#         print(vel)
#
#     # ----------------- RV correct the wavelength array
#     # If the corrected velocity is too large, assume an error occurred and make no change
#     if np.abs(vel) > v_tol:
#         print('corrected velocity too large')
#         corr_wave = np.copy(t_w)
#     # If the radial velocity is small, do the shift
#     else:
#         corr_wave = np.copy(t_w) * (1.0 + (vel / s_o_l))
#
#     # ----------------- Plot the shifted spectra
#     if plot_shift:
#         shifted_f = np.roll(t_f, pix_shift)
#         plt.figure(figsize=(8, 5))
#         plt.plot(t_w, new_tsf, color='tab:orange', linewidth=0.5, label='Template')
#         plt.plot(t_w, t_f, color='k', linestyle=':', linewidth=0.5, label='Original Obs')
#         plt.plot(t_w, shifted_f, linewidth=0.5, label='Corrected Obs: Shifted flux')
#         plt.plot(corr_wave, t_f, linewidth=0.5, label='Corrected Obs: Corrected Wavelengths')
#         plt.legend()
#         plt.title('plot_shift=True\nShifted Spectrum')
#         plt.xlabel(r'Wavelength ($\AA$)')
#         plt.ylabel('Normalized Flux')
#         plt.show()
#
#     # ----------------- Return
#     if return_corr_wave:
#         return corr_wave, vel
#     else:
#         return vel
#

# # -----------------------------------------------------------------------------------------------------------------------
# def doppler_corr(waves, fluxes, s_waves, s_fluxes, plot_shift=False):
#     """
#     Apply a Doppler correction to the given spectrum
#
#     Inputs
#     -------
#     waves: An array of wavelengths for the observed spectrum
#     fluxes: The fluxes corresponding to the wavelengths in wave
#
#     s_waves: An array of wavelengths for the synthetic / reference spectrum
#     s_fluxes: The fluxes corresponding to the wavelengths in s_waves
#
#     plot_shift: If True plot the shifted spectrum for each window (chopped section of spectrum)
#     """
#
#     # ----------------- For each chunk of spectrum...
#     vels = []
#
#     for i in range(len(waves)):
#         # Give the chunck of observed spectrum and synthetic spectrum to pyfxcor to compute the radial velocity
#         # Set the v_tol very high to accept any radial velocity as a solution
#         vel = pyfxcor(waves[i], fluxes[i], s_waves[i], s_fluxes[i], v_tol=1e99, plot_shift=plot_shift)
#         vels.append(vel)
#
#     # Calculate the difference between the velocities
#     v_min = np.min(vels)
#     v_max = np.max(vels)
#
#     delta_v = v_max - v_min
#
#     # If delta v is too large...
#     if delta_v > 5.0:
#         print('Oh shit! Your velocities differ by >5km/s!')
#         print('Measured Velocities: %s' % np.around(np.array(vels), 2))
#
#     return np.array(vels)
#

# # -----------------------------------------------------------------------------------------------------------------------
# def rvcor(spectra,
#           obs_path,
#           synth_spec,
#           synth_path,
#           names,
#           ftype_synth=None,
#           rv_elems='all',
#           manual_rv=None,
#           save_out_spec=True,
#           out_path=None,
#           plot_synth=False,
#           plot_obs=False,
#           plot_regions=False,
#           plot_shift=False,
#           plot_corr=True,
#           print_spec_info=False,
#           print_info=False,
#           save_rv_info=False,
#           rv_save_name=None,
#           print_corr_spec_info=False,
#           save_plot=False,
#           save_plot_path=None,
#           pause=False):
#     """
#     Radial Velocity Corrections
#
#     Inputs
#     -------
#
#     spectra: A list of spectra to which to apply the radial velocity correction
#
#     obs_path: The path to the folder which contains the spectra
#
#     syth_spec: The name of the template or synthetic spectrum
#
#     template_path: The path to the template or synthetic spectrum
#
#     names: Names of the objects corresponding to the spectra
#
#     rv_elems: Elements to do the radial velocity corrections with
#     """
#
#     if out_path is None:
#         out_path = obs_path
#
#     if save_plot_path is None:
#         save_plot_path = obs_path
#
#     if rv_elems == 'all':
#         rv_elems = ['Mg', 'CaT', 'Ha', 'Hb', 'Hepsilon', 'Hgamma', 'Hdelta', 'SrII_4215.524',
#                     'BaII_4554.030', 'FeII_4923.922', 'FeI_4957.596', 'FeII_5018.435', 'FeI_5269.537', 'CaI_6122.217']
#
#     # ----------------- Read in synthetic spectra
#
#     # -------- Automatic Detect File Type
#     if ftype_synth is None:
#         ftype_synth = synth_spec.split('.')[1]
#
#     s_wave, s_flux, *s_err = read_spec(synth_path + synth_spec, ftype=ftype_synth)
#
#     # ----------------- Chop the synthetic spectrum
#     '''Isolate and keep only sections of the synthetic spectrum around the lines
#
#         - s_waves is an array with a certain number of sub arrays (= to the number of elements chosen).  Each
#         sub-array corresponds to the spectrum chopped around an element. '''
#     s_waves, s_fluxes = spec_chop(s_wave, s_flux, rv_elems)
#
#     # ----------------- Plot the synthetic (reference) spectra
#     if plot_synth is True:
#         plt.figure(figsize=(8, 5))
#         plt.plot(s_wave, s_flux, color='tab:orange', linewidth=0.5, label='Synthetic')
#         for i in range(len(s_waves)):
#             plt.plot(s_waves[i], s_fluxes[i], linewidth=0.5, label=rv_elems[i])
#         plt.legend()
#         plt.xlabel(r'Wavelength ($\AA$)')
#         plt.ylabel('Normalized Flux')
#         plt.title('plt_synth=True\nSynthetic Template')
#         plt.show()
#         if save_plot is True:
#             plt.savefig(save_plot_path + 'SyntheticTemplate.eps', format='eps')
#     # -----------------
#
#     # ----------------- Examine (observed) Spectra
#     if type(spectra) == str:
#         spectra = [spectra]
#
#     if type(names) == str:
#         names = [names]
#
#     best_vs = []
#     best_vs_err = []
#
#     # ----------------- For each (observed) spectrum:
#     for n, s in enumerate(spectra):
#         print('')
#         print('Correcting ', names[n])
#
#         # ----------------- Automatic Detect File Type
#         ftype = s.split('.')[1]
#         if ftype not in ['bin', 'xyz']:
#             print('Data type must be .bin or .xyz\nSkipping {},{}'.format(n, s))
#             continue
#
#         # ----------------- Read in (observed) spectra
#         spectrum = read_spec(os.path.join(obs_path, s), ftype=ftype)
#         wave = spectrum[0]
#         flux = spectrum[1]
#         err = spectrum[2]
#
#         # ----------------- Chop the (observed) spectrum
#         '''Isolate and keep only sections of the synthetic spectrum around the Mg b lines (5167A, 5172A, 5183A),
#         the Na D lines (5896Å, 5890Å), and/or the Calcium Triplet (8498A, 8542 A, 8662 A)
#
#         - waves is an array with a certain number of sub arrays (= to the number of elements chosen).  Each sub-array
#         corresponds to the spectrum chopped around an element. '''
#         waves, fluxes = spec_chop(wave, flux, rv_elems)
#
#         # ----------------- Print information about the (observed) spectrum:
#         if print_spec_info is True:
#             print(names[n])
#             for i in range(len(waves)):
#                 print('Length of %s obs spec: ' % rv_elems[i], len(fluxes[i]))
#                 print('Length of %s obs wave: ' % rv_elems[i], len(waves[i]))
#                 print('%s obs start: ' % rv_elems[i], np.min(waves[i]))
#                 print('%s obs end: ' % rv_elems[i], np.max(waves[i]))
#                 print('Length of %s synth spec: ' % rv_elems[i], len(s_fluxes[i]))
#                 print('Length of %s synth wave: ' % rv_elems[i], len(s_waves[i]))
#                 print('Synth %s start: ' % rv_elems[i], np.min(s_waves[i]))
#                 print('Synth %s end: ' % rv_elems[i], np.max(s_waves[i]))
#                 print('---')
#         # -----------------
#
#         # ----------------- Plot the (observed) spectrum:
#         if plot_obs is True:
#             plt.figure(figsize=(8, 5))
#             plt.plot(wave, flux, color='k', linewidth=0.5, label=names[n])
#             for i in range(len(waves)):
#                 plt.plot(waves[i], fluxes[i], linewidth=0.5, label=rv_elems[i])
#             plt.legend()
#             plt.xlabel(r'Wavelength ($\AA$)')
#             plt.ylabel('Normalized Flux')
#             plt.title('plot_obs=True\nGRACES: %s' % names[n])
#             plt.show()
#             if save_plot is True:
#                 plt.savefig(save_plot_path + '%s_GRACES.eps' % names[n], format='eps')
#
#         # ----------------- Plot the "windows" around the specified elements for the (observed) spectrum:
#         if plot_regions is True:
#             for i in range(len(waves)):
#                 plt.figure(figsize=(8, 5))
#                 plt.plot(waves[i], fluxes[i], color='k', linewidth=0.5, label=names[n])
#                 plt.plot(s_waves[i], s_fluxes[i], color='tab:orange', linewidth=0.5, label='Template')
#                 if rv_elems[i] == 'Mg':
#                     plt.xlim(5050, 5300)
#                 if rv_elems[i] == 'Na':
#                     plt.xlim(5800, 5990)
#                 if rv_elems[i] == 'Ca':
#                     plt.xlim(8400, 8750)
#                 if rv_elems[i] == 'Ha':
#                     plt.xlim(6520, 6600)
#                 if rv_elems[i] == 'Hb':
#                     plt.xlim(4820, 4900)
#
#                 if rv_elems[i] == 'Mg_Tell':
#                     plt.xlim(4420, 4500)
#                 if rv_elems[i] == 'O_Tell':
#                     plt.xlim(7750, 7800)
#
#                 plt.legend()
#                 plt.xlabel(r'Wavelength ($\AA$)')
#                 plt.ylabel('Normalized Flux')
#                 plt.title('plot_regions=True\nGRACES %s lines: %s' % (rv_elems[i], names[n]))
#                 plt.show()
#                 if save_plot is True:
#                     plt.savefig(save_plot_path + '%s_%s_lines.eps' % (names[n], rv_elems[i]), format='eps')
#
#         # ----------------- Automatic rv correction
#         ''' If  manual_rv is not selected, then do an automatic rv correction
#         '''
#         if manual_rv is None:
#             vels = doppler_corr(waves, fluxes, s_waves, s_fluxes, plot_shift)
#
#             # Calculate the average velocity end error from each line
#             best_v = np.mean(vels)
#             best_v_err = np.std(vels)
#
#         # ----------------- Manual rv correction
#         else:
#             '''If things are wonky, accept a manual input for the RV correction and overwrite some stuff (to be
#             expanded). Calculate the average RV from the three regions (error bar????) '''
#             vels = manual_rv
#             best_v = vels[3]
#             best_v_err = float(np.std(vels[:-1]))
#         # -----------------
#
#         best_vs.append(best_v)
#         best_vs_err.append(best_v_err)
#
#         # ----------------- Print information
#         if print_info is True:
#             if manual_rv is None:
#                 for i in range(len(rv_elems)):
#                     print('Velocity correction from %s lines: ' % rv_elems[i], vels[i])
#             else:
#                 print('By hand velocity from Mg lines: ', vels[0])
#                 print('By hand velocity from Na lines: ', vels[1])
#                 print('By hand velocity from Ca triplet: ', vels[2])
#                 print('By hand velocity from Ha: ', vels[3])
#                 print('By hand velocity from Hb: ', vels[4])
#                 print('By hand velocity from Mg_Tell: ', vels[5])
#                 print('By hand velocity from O_Tell: ', vels[6])
#
#             print(r'Mean radial velocity: %s $\pm$ %s' % (round(best_v, 2), round(best_v_err, 2)))
#
#         # ----------------- RV correct the wavelength array
#         '''
#         Apply the average radial velocity shift to the entire original wavelength array
#         '''
#         corr_wave = np.copy(wave) * (1.0 + best_v / s_o_l)
#
#         # -----------------  Interpolate the original flux onto the corrected wavelength array
#         f = interpolate.interp1d(corr_wave, np.copy(flux), kind='cubic', bounds_error=False, fill_value=1.0)
#         ef = interpolate.interp1d(corr_wave, np.copy(err), kind='cubic', bounds_error=False, fill_value=1.0)
#
#         # ----------------- Transpose the corrected flux array back onto the original grid
#         corr_flux = f(np.copy(wave))
#         corr_err = ef(np.copy(wave))
#
#         # -----------------
#         if plot_corr is True:
#             plt.figure(figsize=(8, 5))
#             plt.plot(s_wave, s_flux, color='tab:orange', linewidth=0.5, label='Template')
#             plt.plot(wave, flux, color='k', linestyle='-', linewidth=0.5, label='Original Obs')
#             plt.plot(wave, corr_flux, color='tab:green', linestyle=':', linewidth=0.5, label='Corrected Obs')
#             plt.plot(wave, err, color='tab:red', linestyle='-', linewidth=0.5, label='Original Err')
#             plt.plot(wave, corr_err, color='tab:blue', linestyle=':', linewidth=0.5, label='Corrected Err')
#             plt.legend()
#             # plt.xlim(8400,8750)
#             plt.ylim(0, 1.5)
#             plt.xlabel(r'Wavelength ($\AA$)')
#             plt.ylabel('Normalized Flux')
#             plt.title('plot_corr=True\nRV Corrected GRACES: %s' % names[n])
#             plt.show()
#             if save_plot is True:
#                 plt.savefig(save_plot_path + '%s_RV_Corr.eps' % names[n], format='eps')
#
#         # -----------------
#         if print_corr_spec_info is True:
#             dl = []
#             for i in range(len(wave) - 1):
#                 dl.append(wave[i + 1] - wave[i])
#             dl_best = np.mean(dl)
#             print('Starting wavelength: ', np.min(wave))
#             print('Ending wavelength: ', np.max(wave))
#             print('Min delta lambda: ', np.min(dl))
#             print('Max delta lambda: ', np.max(dl))
#             print('Mean delta lambda: ', dl_best)
#             print('Length of wavelength array: ', len(wave))
#             print('Length of corrected flux array: ', len(corr_flux))
#
#         if save_out_spec is True:
#             colheads = ['wave', 'flux', 'err']
#             table = Table([wave, corr_flux, corr_err], names=colheads)
#             # pickle.dump(table,open(out_path+names[n]+'.bin','wb'))
#             print('Saving radial velocity corrected spectrum to ' + out_path + s[:-4] + '.rv.bin')
#             pickle.dump(table, open(out_path + s[:-4] + '.rv.bin', 'wb'))
#
#         if pause is True:
#             input("Press Enter to continue to the next star...")
#
#     # -----------------
#     if save_rv_info is True:
#         best_vs = -1 * np.array(best_vs)
#         colheads = ['name', 'mean_v', 'mean_v_err'] + [rv_elems[i] for i in range(len(rv_elems))]
#         tab = [names, best_vs, best_vs_err] + [[vels[i]] for i in range(len(rv_elems))]
#
#         table = Table(tab, names=colheads)
#         # ascii.write(table,'GRACESRVs.txt')
#         print('Saving radial velocity to ' + rv_save_name)
#         ascii.write(table, rv_save_name, overwrite=True)
#         # pickle.dump(table,open(out_path+name[n]+'.bin','wb'))
#

# -----------------------------------------------------------------------------------------------------------------------
def order_cut(spectra, spath, opath=None, save=False, save_locs=False, ftype='fits'):
    """
    A function to correct order overlap (A Standardize OPERA function)
    ----------

    "order_cut" identifies where the spectral arrays 'double back" (i.e. where we jump from the red end of echelle
    order N to the blue end of echelle order N + 1) and combines those two regions via a weighted average using the
    i.fits "error spectrum" to assign the weights.

This increases the SNR in those regions and makes the spectral arrays all nice and linear.

Input are:

"files" - A list or array of files to examine

"spath" - Path to the directory containing said files

"opath" - Path to the directory where the corrected spectra should be saved

"save" - If True, save the order-corrected spectrum

"save_locs" - If True, save the locations of the order overlaps
    """
    if ftype not in ['fits', 'bin']:
        raise TypeError('ftype must be fits or bin')

    if type(spectra) == str:
        spectra = [spectra]

    olaps = []

    # ----------------- Examine each spectrum
    # For each spectrum:
    for s in spectra:
        # Get wavelength, flux, and error information

        wave, flux, err = read_spec(os.path.join(spath, s), ftype=ftype)

        olap_locs = []

        # -----------------  Iterate through wavelength point of the spectrum
        i = 0
        while i < len(wave) - 1:

            # -----------------  Find the end of the Echelle order
            '''If the wavelength of the next index over is less than the wavelength of the current index, we must be 
            at the end of an Echelle order '''
            if wave[i + 1] < wave[i]:

                # The location where the order stops:
                stop1 = [i, wave[i]]  # End of the (N)th order
                # The location where the next order starts:
                start2 = [i + 1, wave[i + 1]]  # Start of (N+1)th order

                # ----------------- Find the order overlap
                '''Find the region where the (N)th order and (N+1)th order 
                have data for the same wavelengths    
                '''

                '''In the region before the order split, find where the wavelength is closest to the wavelength at 
                the start of the order (N+1) '''
                start1 = find_nearest(wave[:i], start2[1])

                '''In the region after the order split, find where the wavelength is closest to the wavelength at the 
                end of the order (N). The index of sp2 is in terms of the array wave[i+1:], not the array "wave", 
                so translate the index into an index of "wave" '''
                sp2 = find_nearest(wave[i + 1:], stop1[1])
                stop2 = (i + 1 + sp2[0], sp2[1])

                # ----------------- Extract the wavelengths, fluxes, and errors of the overlap
                olap1_w = wave[start1[0]: stop1[0] + 1]
                olap2_w = wave[start2[0]: stop2[0] + 1]

                olap1_f = flux[start1[0]: stop1[0] + 1]
                olap2_f = flux[start2[0]: stop2[0] + 1]

                olap1_e = err[start1[0]: stop1[0] + 1]
                olap2_e = err[start2[0]: stop2[0] + 1]

                # ----------------- Determine the new start and stop points for the interpolated overlap
                '''Note, the wavelengths of the start of the (N)th order and the start of the (N+1)th order are not 
                the same (this goes for the end wavelengths as well).  One set of start/stop wavelengths need to be 
                chosen '''
                new_stt = np.min([start1[1], start2[1]])
                new_start = [start1[0], new_stt]

                new_stp = np.max([stop1[1], stop2[1]])
                new_stop = [stop2[0], new_stp]

                # ----------------- Determine the new dispersion by a bunch of averages over the two overlapping regions
                '''
                Find the mean difference in the (m)th wavelength and (m+1)th wavelength
                '''
                dw = []
                for i, w in enumerate(olap1_w):
                    if i < len(olap1_w) - 1:
                        dw.append(olap1_w[i + 1] - olap1_w[i])
                dw1 = np.mean(dw)
                dw = []
                for i, w in enumerate(olap2_w):
                    if i < len(olap2_w) - 1:
                        dw.append(olap2_w[i + 1] - olap2_w[i])
                dw2 = np.mean(dw)
                dw = np.mean([dw1, dw2])

                # ----------------- Create the new wavelength array for the overlapped region using the dispersion
                # found above
                N = int(round((new_stop[1] - new_start[1]) / dw))  # Number of steps
                i_w = np.linspace(new_start[1], new_stop[1], N)  # New wavelength

                # ----------------- Save the cut boundaries for later use
                if save_locs:
                    olap_locs.append([new_start[1], new_stop[1]])

                # ----------------- Do the interpolation
                '''Interpolate the flux and error data for the overlapped part of the (N)th and (N+1)th orders onto 
                the same (new) wavelength array '''
                i_f1 = np.interp(i_w, olap1_w, olap1_f)
                i_e1 = np.interp(i_w, olap1_w, olap1_e)

                i_f2 = np.interp(i_w, olap2_w, olap2_f)
                i_e2 = np.interp(i_w, olap2_w, olap2_e)

                # ----------------- Perform the weighted average on the interpolated orders
                new_flux, new_err = wavg(np.array([i_f1, i_f2]), np.array([i_e1, i_e2]))

                # ----------------- Replace the sections in the arrays, temporarily mask the gap regions
                wave[new_start[0]:new_stop[0] + 1] = -999.
                wave[new_start[0]: new_start[0] + len(i_w)] = i_w

                flux[new_start[0]:new_stop[0] + 1] = -999.
                flux[new_start[0]: new_start[0] + len(i_w)] = new_flux

                err[new_start[0]:new_stop[0] + 1] = -999.
                err[new_start[0]: new_start[0] + len(i_w)] = new_err

                i = new_stop[0]

            else:
                i += 1

        gwave = list(wave)
        gflux = list(flux)
        gerr = list(err)

        gwave = [w for w in gwave if w != -999.]
        gflux = [f for f in gflux if f != -999.]
        gerr = [e for e in gerr if e != -999.]

        # gdata = [gwave, gflux, gerr]

        # ----------------- Save new order-corrected spectrum
        if save is True:
            print('Saving order corrected spectrum to ' + opath + s[:-6] + '.o.bin')

            colheads = ['wave', 'flux', 'err']
            table = Table([gwave, gflux, gerr], names=colheads)

            if opath is None:
                opath = spath

            # ascii.write(table, opath + s[:-6] + '.o')
            pickle.dump(table, open(opath + s[:-6] + '.o.bin', 'wb'))

        # ----------------- Save locations of the order overlaps
        if save_locs:
            olaps.append(olap_locs + [s])

    # ----------------- Save file of locations of the order overlaps
    if save_locs:
        print('Saving locations of order overlaps to Standards_Olap_Info.npy')
        np.save(os.path.join(os.getcwd(), 'Standards_Olap_Info.npy'), np.array(olaps, dtype=object))


# -----------------------------------------------------------------------------------------------------------------------
def order_chop(wave, flux, err, w_lo, w_hi):
    """
    Chop a spectrum to the specified wavelength range

    Inputs
    -------

    wave: wavelength array

    flux: the flux array corresponding to the wavelength array

    err: the error arrray corresponding to the wavelength array

    w_lo: the lower limit of the wavelength range

    w_hi: the upper limit of the wavelength range
    """
    wave = np.array(wave)

    good = np.where((wave >= w_lo) & (wave <= w_hi))

    short_wave = wave[good]
    short_flux = flux[good]
    short_err = err[good]

    return short_wave, short_flux, short_err


# -----------------------------------------------------------------------------------------------------------------------
def find_late_gaps(wave, flux, last_order):
    """
    Find the starting and ending wavelengths of the wavelength gaps that occur in the spectra at high Echelle orders
    (A Standardize OPERA function)

    Inputs
    -------

    wave: wavelength array

    flux: the flux array corresponding to the wavelength array

    last_order: the ending wavelength of the last Echelle order that overlaps with a previous Echelle order
    """
    good = np.where(np.asarray(wave) >= last_order)[0]

    short_wave = wave[good]
    short_flux = flux[good]

    i = 0
    cntr = 0
    start = 0

    gaps = []

    while i < len(short_wave) - 2:

        # ----------------- Calculate the slope between the 0th and 1st elements
        dw1 = short_wave[i + 1] - short_wave[i]
        df1 = short_flux[i + 1] - short_flux[i]
        slope1 = df1 / dw1

        # ----------------- Calculate the slope between the 1st and 2nd elements
        dw2 = short_wave[i + 2] - short_wave[i + 1]
        df2 = short_flux[i + 2] - short_flux[i + 1]
        slope2 = df2 / dw2

        # ----------------- Calculate the difference between the slopes
        diff = np.abs(slope2 - slope1)

        # -----------------
        # If the difference is less than the threshold (chosen by eye)
        if diff < 5E-4:
            # If this is the first time the difference has been small,
            # record the starting index
            if cntr == 0:
                start = i

            cntr += 1
            i += 1
        # -----------------
        else:
            # If the difference has been small but is no longer...
            # And if this has happened at least 5 times (avoid local coincidences)
            if cntr >= 5.:
                # Save the indices
                gaps.append([short_wave[start], short_wave[i]])
            cntr = 0
            i += 1

    return gaps


# -----------------------------------------------------------------------------------------------------------------------
def find_order_splits(wave, flux, orders):
    """
    Find the locations where orders change (A Standardize OPERA function)

    Inputs
    -------

    wave: wavelength array

    flux: the flux array corresponding to the wavelength array

    orders: An array containing the mean start and stop wavelength of each order
    """

    late_gaps = find_late_gaps(wave, flux, orders[-1][1])

    first = [np.min(wave), orders[0][0]]

    good_order = [first] + orders

    good_order.append([orders[-1][1], late_gaps[0][0]])

    for i in range(len(late_gaps) - 1):
        good_order.append([late_gaps[i][1], late_gaps[i + 1][0]])

    good_order.append([late_gaps[-1][1], np.max(wave)])

    return good_order


# -----------------------------------------------------------------------------------------------------------------------
def contnorm_sigclip(y, sigma_lower, sigma_upper):
    """
    Improve the continuum position with asymetric sigma-clipping
    ------------
    Best For: Spectra which have already been continuum normalized such that
    the continuum is already horizontal. This routine will bring the continuum
    value closer to 1.

    - New continuum normalization code (written by J. Glover)

    :param y: (list or array) Flux values
    :param sigma_lower: (float) Lower sigma value for clipping
    :param sigma_upper: (float) Upper sigma value for clipping

    Returns: normalized flux array
    """

    # Sigma-clip to estimate the continuum value
    sol = sigma_clip(y, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)
    clip_norm = [y[i] for i in range(len(sol.mask)) if sol.mask[i] == False]

    # Divide flux by the average estimated continuum value to bring
    # the continuum closer to 1
    norm = y / np.mean(clip_norm)

    return norm


# -----------------------------------------------------------------------------------------------------------------------
def contnorm_filter(y, smooth_kernel, mode='reflect'):
    """
    Continuum Normalization by estimating the continuum with a filter
    ------------
    Best For: "Raw" spectra which have not been continuum normalized yet.

    - New continuum normalization code (written by J. Glover)

    :param y: (list or array) Flux values
    :param smooth_kernel: (int) Length of the kernel in units of array indices (pix)
    :param mode: (float) Descirbes how the input array is extended beyond its boundaries

    returns: normalized flux array, filtered flux array
    """

    filt = median_filter(y, size=smooth_kernel, mode=mode)

    # To avoid divide by zero errors, wherever the filter = 0, set the filter to 1
    ind = np.where(filt == 0)
    filt[ind] = 1

    norm = y / filt

    return norm, filt


# -----------------------------------------------------------------------------------------------------------------------
def contnorm_2stage(y, smooth_kernel, sigma_lower, sigma_upper, mode='reflect'):
    """
    Two-Stage Continuum Normalization Routine
    ------------
    Best For: "Raw" spectra which have not been continuum normalized yet.

    - New continuum normalization code (written by J. Glover)

    Stage 1: Median Filter Normalization

    Find the first estimate of the continuum by applying a
    median filter with width = smooth_kernel to the spectrum.
    The spectrum is normalized by dividing by the filtered spectrum.
        - Your results will be affected by the size of smooth_kernel.
        A kernel with a width that is a significant fraction of the total length of the spectrum
        is a good place to start ( ie smooth_kernel = len(y)/2 )

    Stage 2: k-sigma clipping Normalization

    Find the second estimate of the continuum by applying a k-sigma clipping routine
    to the normalized spectrum from stage 1. Normalize the spectrum a second time by dividing by the
    mean of the continuum.
        - Your results will be affected by the choice of the upper and lower sigma limits.
        simga_lower = 1.5 and sigma_upper = 3 is good place to start

    Returns
    -----
    norm2: (array) The continuum-normalized spectrum
    filt: (array) The filtered spectrum. Useful for checking and adjusting the smooth_kernel size
    """

    # ------- Step 1: Smooth the flux to get the first estimate of the continuum, then norm
    norm, filt = contnorm_filter(y, smooth_kernel, mode=mode)

    # ------ Step 2: Improve the estimate of the continuum. Sigma clip the normalized flux. Then norm again
    norm2 = contnorm_sigclip(norm, sigma_lower, sigma_upper)

    return norm2, filt


# -----------------------------------------------------------------------------------------------------------------------
class drawContinuum:
    """
    Estimate the continuum function via user input and perform continuum normalization
    ---------------
    Connects to a matplotlib figure via matplotlib event handling and allows user to draw a continuum by placing
     points on the given figure.
    - Click on the spectrum to place a point on the continuum. Right click that point to delete it

    Best For: "Raw" spectra which have not been continuum normalized yet, especially those with difficult and/or
    unusual continuums. For example, spectra which contain deep and wide spectral lines.

    - New continuum normalization code (written by J. Glover)

    Parameters
    ----
    :param w: (list or array) wavelength array
    :param f: (list or array) flux array corresponding to w

    Attributes
    ----
    self.cont: list of continuum values (corresponds to same wavelength grid as w)
    self.norm: list of normalized flux (corresponds to same wavelength grid as w)


    Example Useage
    ----
    %matplotlib notebook
    import matplotlib.pyplot as plt
    import numpy as np
    import asap_lib.spectra as sa

    # ---------- Initialize and create continuum object
    continuum = sa.drawContinuum(w, f)
    continuum.connect()

    #
    #  ... User places points along the continuum ...
    # (Tip: Use continuum.Help() to get a list of commands)

    # ---------- Normalize
    continuum.disconnect()
    continuum.normalize()

    # ---------- Save the continuum normalized spectrum
    sa.write2bin(w, continuum.norm, err, save_path )

    """

    def __init__(self, w, f):

        fig = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)
        plt.subplots_adjust(hspace=0.5)

        self.ax1 = ax1
        self.ax2 = ax2

        self.fig = fig
        self.wave = w
        self.flux = f

        self.norm = None
        self.cont = None
        self.cid_click = None
        self.cid_pick = None
        self.cid_key = None

        self.reference = False
        self.ref_w = None
        self.ref_f = None

        # ----- Plot
        self.ax1.set_title('Original Spectrum')
        self.ax2.set_title('Normalized Spectrum')
        self.fig.supxlabel('Wavelength')
        self.fig.supylabel('Flux')
        self.ax1.plot(self.wave, self.flux, '-', picker=5)  # 5 points tolerance
        self.xlim = copy.copy(self.ax1.get_xlim())
        self.ylim = copy.copy(self.ax1.get_ylim())

        self.ax2.set_ylim(0, 1.5)
        self.ax2.hlines(1, self.xlim[0], self.xlim[1], alpha=0.25, color='grey', zorder=0)

    def connect(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_pick = self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.ontype)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_pick)
        self.fig.canvas.mpl_disconnect(self.cid_key)

    def Help(self):
        print('- Left click on the original spectrum (upper plot) to place points along the continuum ')
        print('- Right click on the original spectrum (upper plot) to delete continuum points ')
        print('- Continuum points cannot be placed when a toolbar item is active ')
        print('- Hit h to return to the original view ')
        print('- Hit r to clear all points ')
        print('- Hit enter to create the continuum and normalize with it ')

    def addReference(self, ref_w, ref_f):
        self.reference = True
        self.ref_w = ref_w
        self.ref_f = ref_f

        self.ax2.plot(self.ref_w, self.ref_f, color='grey', alpha=0.5, lw=2, zorder=0)

    # --------- Define Valid Events

    # --- Place continuum points (do not place points if a toolbar button is active)
    def onclick(self, event):
        # place a scatter point on the cursor location (as long as the cursor is within subplot1)
        toolbar = plt.get_current_fig_manager().toolbar
        if event.inaxes == self.ax1:
            if event.button == 1 and toolbar.mode == '':
                self.ax1.plot(event.xdata, event.ydata, 'ro', picker=5, label='cont_pnt')

    # --- Select and remove continuum points with a right click
    def onpick(self, event):
        if event.mouseevent.button == 3:
            if hasattr(event.artist, 'get_label') and event.artist.get_label() == 'cont_pnt':
                event.artist.remove()

    # --- Do stuff on key press
    def ontype(self, event):

        # If the pressed key was "r"
        if event.key == 'r':
            # Clear axis 1 and replot the original
            plt.sca(self.ax1)
            plt.cla()
            self.ax1.plot(self.wave, self.flux, '-', picker=5)  # 5 points tolerance
            self.ax1.set_title('Original Spectrum')
            self.fig.supxlabel('Wavelength')
            self.fig.supylabel('Flux')

            # Clear axis 2 and replot original
            plt.sca(self.ax2)
            plt.cla()
            self.ax2.set_title('Normalized Spectrum')
            self.ax2.set_ylim(0, 1.5)
            self.ax2.hlines(1, self.xlim[0], self.xlim[1], alpha=0.25, color='grey', zorder=0)

        # If the pressed key was "h"
        if event.key == 'h':
            self.ax1.set_xlim(self.xlim[0], self.xlim[1])
            self.ax1.set_ylim(self.ylim[0], self.ylim[1])

            # If the key was "enter" connect the points by interpolating between them
        if event.key == 'enter':

            plt.sca(self.ax1)

            cont_pnt_coord = []

            for artist in plt.gca().get_children():
                if hasattr(artist, 'get_label') and artist.get_label() == 'cont_pnt':
                    cont_pnt_coord.append(artist.get_data())
                elif hasattr(artist, 'get_label') and artist.get_label() == 'continuum':
                    artist.remove()

            cont_pnt_coord = np.array(cont_pnt_coord)[..., 0]
            sort_array = np.argsort(cont_pnt_coord[:, 0])
            x, y = cont_pnt_coord[sort_array].T
            self.cont = np.interp(self.wave, x, y)
            self.ax1.plot(self.wave, self.cont, 'r--', lw=2, label='continuum')

            plt.sca(self.ax2)
            self.norm = self.flux / self.cont
            plt.cla()
            self.ax2.set_title('Normalized Spectrum')
            self.ax2.plot(self.wave, self.norm, '-g')
            self.ax2.set_ylim(0, 1.5)
            self.ax2.hlines(1, self.xlim[0], self.xlim[1], alpha=0.25, color='grey', zorder=0)
            if self.reference:
                self.ax2.plot(self.ref_w, self.ref_f, color='grey', alpha=0.5, lw=2, zorder=0)


# -----------------------------------------------------------------------------------------------------------------------
def sig_clip(wave, flux, clo, chi, window, step, reference):
    """
    Original Continuum Position Adjustment via sigma-clipping (written by C. Kielty)
    -------------------
    Perform sigma-clipping on the provided spectrum or spectrum segment

    Best For: Standardize OPERA

    Inputs
    ------

    wave: the wavelength array

    flux: the flux array corresponding to the wavelength array

    clo: the lower sigma-cipping limit

    chi: the upper sigma-clipping limit

    window: Size of the sliding boxar in pixels; needs to be an odd-number

    step: some value that the Savitzky-Golay filter uses

    reference: reference flux value for continuum
    """

    # ----------------- Clip the flux array
    '''
    Sigma clip the spectrum (flux) so we don't try to fit noise. 
    c_flux is the flux array with the clipped flux values removed. 
    min_t is the lower flux threshold, and max_t is the upper flux threshold that spst.sigmaclip used in clipping.
    '''
    c_flux, min_t, max_t = spst.sigmaclip(flux, low=clo, high=chi)

    # ----------------- Extract the spectrum data that is within the clipping threshold values
    good = np.where((flux >= min_t) & (flux <= max_t))[0]
    c_wave = wave[good]

    # ----------------- Check that the window is okay, if not, shorten it to an okay length
    if len(c_wave) <= window:
        close = len(c_wave - 10)
        window = int(np.ceil(close) // 2 * 2 + 1)

    # ----------------- Rather than fitting a polynomial, heavily smooth the spectrum and fit to the smoothed curve
    filt = spsi.savgol_filter(c_flux, window, step)

    # ----------------- Interpolate the filtered data back onto the original wavelength grid
    f = interp1d(c_wave, filt, bounds_error=False, fill_value='extrapolate')
    curve = f(wave)

    # -----------------
    '''
    Shift the polynomial by the standard deviation of the flux array. 
    The goal here is to set the continuum closer to 1.0 (rather than the mean of the spectrum)
    '''
    cont = curve + np.std(c_flux)

    median = np.median(cont)
    scale = reference / median
    cont = scale * cont

    return c_wave, c_flux, curve, cont


# -----------------------------------------------------------------------------------------------------------------------
def spec_norm(spectra, spath,
              order_info,
              clips, window, step, reference,
              plot_orig=False,
              plot_clip=False,
              plot_norm=False,
              save_norm=False,
              out_path=None):
    """
    Original Spectrum Continuum Normalization (written by C. Kielty)
    -------------------
    Normalize spectra with a sliding box asymmetric clipping routine

    Best For: Standardize OPERA

    Inputs:
    --------

    spectra: A list of spectra to normalize
    spath: The path to the spectra to normalize
    order_info: A file containing some information where order overlaps roughly occur
    clips: [lower, upper] the lower and upper sigma to clip
    window: Size of the sliding boxcar in pixels; needs to be an odd-number
    step: some value that the Savitzky-Golay filter uses
    reference: reference flux value for continuum

    """

    # ----------------- Read in information about where the orders overlap
    '''
    Earlier a file containing information about order overlaps was generated.  
    '''
    orders = np.load(order_info, allow_pickle=True)
    olaps = []
    for o in orders:
        stop = [o[-1]]
        for i in range(len(o) - 2):
            stop.append([o[i][1], o[i + 1][1]])
        olaps.append(stop)

    # For each order overlap find the mean start and stop wavelength of that order
    orders = []
    for i in range(1, len(olaps[0])):
        sts = []
        sps = []
        for o in olaps:
            sts.append(o[i][0])
            sps.append(o[i][1])
        start = np.mean(sts)
        stop = np.mean(sps)
        pair = [start, stop]
        orders.append(pair)

    # ----------------- For each spectrum
    for spectrum in spectra:

        # ----------------- Read in the spectrum
        data = read_spec(os.path.join(spath, spectrum), ftype='bin')
        wave = data[0]
        flux = data[1]
        err = data[2]

        # ----------------- Plot Original Spectrum
        if plot_orig:
            fig1 = plt.figure(figsize=(8, 5))
            ax1 = fig1.add_subplot(111)
            ax1.plot(wave, flux, color='k', linewidth=0.5)
            ax1.set_title('plot_orig=True\nOriginal Spectrum and Orders')
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')
            ax1.set_ylim(-5, 5)

        # ----------------- Plot Clipped Spectrum
        if plot_clip:
            fig2 = plt.figure(figsize=(8, 5))
            ax2 = fig2.add_subplot(111)
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')

        # ----------------- Using the mean order overlap start and stop, find the start and stop of each order
        good_order = find_order_splits(wave, flux, orders)

        # ----------------- Output arrays for the normalized spectrum
        flat_flux = np.array([])
        flat_wave = np.array([])
        flat_err = np.array([])

        # -----------------  For each order in the spectrum...
        for i in range(1, len(good_order)):

            # ----------------- Plot Original
            if plot_orig:
                # If plotting, highlight where the order overlaps occur
                ax1.axvline(good_order[i][0], ls=':', c='g')
                ax1.axvline(good_order[i][1], ls=':', c='r')

            # -----------------  Isolate each order to perform a localized continuum normalization
            short_wave, short_flux, short_err = order_chop(wave, flux, err, good_order[i][0], good_order[i][1])

            # ----------------- Sigma-Clip
            '''
            Using a sigma-clipping routine, clip the short spectrum and use a Savitzky-Golay filter to determine 
            the global structure of the short spectrum. 
            This seems to work better than a polynomial as it handles "jumps" better
            '''
            c_wave, c_flux, curve, cont = sig_clip(short_wave, short_flux, clips[0], clips[1], window, step, reference)

            # ----------------- Plot Clipped Spectrum
            if plot_clip:
                ax2.plot(short_wave, short_flux, color='k', linewidth=0.5, label='Order')
                ax2.plot(c_wave, c_flux, color='tab:green', linewidth=0.5, label='Clipped')
                ax2.plot(short_wave, curve, color='tab:blue', linewidth=0.5, label='Curve')
                ax2.plot(short_wave, cont, color='tab:red', linewidth=0.5, label='Continuum')

            # ----------------- Normalize the order
            # Divide the order by the continuum to normalize that order
            flat = np.array(short_flux / cont)

            # ----------------- Concatenate the order onto the output arrays
            flat_flux = np.concatenate([flat_flux, flat])
            flat_wave = np.concatenate([flat_wave, short_wave])
            flat_err = np.concatenate([flat_err, short_err])

        # ----------------- Plot Clipped Spectrum
        if plot_clip:
            ax2.set_title('plot_clip=True\nOrder by Order Curves: %s' % spectrum)
            ax2.set_ylim(-5, 5)

        # ----------------- Plot Normalized Spectrum
        if plot_norm:
            plt.figure(figsize=(8, 5))
            plt.plot(flat_wave, flat_flux, color='k', linewidth=0.5)
            plt.title('plot_norm=True\nFinal normalized: %s' % spectrum)
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux')
            plt.axhline(1.0, ls=':', c='gray')
            plt.ylim(-5, 5)

        # ----------------- Save Continuum Normalized Spectrum
        if save_norm:
            name = os.path.splitext(spectrum)[0]
            data = [flat_wave, flat_flux, flat_err]
            colheads = ['wave', 'flux', 'err']
            table = Table(data, names=colheads)
            pickle.dump(table, open(out_path + name + '.norm.bin', 'wb'))


# -----------------------------------------------------------------------------------------------------------------------
def find_continuum(obswave,
                   obsflux,
                   linewaves,
                   ref_wave=None,
                   line_width=0.5,
                   wave_range=5.0,
                   trim_sigma=[2.0, 2.0],
                   retrim=True):
    """
    Find wavelength and flux value of the spectrum which constitute the continuum (written by C. Kielty)
    --------------
    A function to identify the wavelengths and fluxes of the continuum near a spectral line.
    This is the old / original version written by Colin Kielty

    :param obswave: (array) Wavelength array of the spectrum

    :param obsflux: (array) Flux array corresponding to the wavelength array of the spectrum

    :param linewaves: (list/array or float) Wavelength(s) of the spectral lines in question

    :param ref_wave:

    :param line_width: (float) Estimated width of the spectral line in wavelength units (angstroms)

    :param wave_range: (float) In the case of a single line, width of window around the line. In the case of several
    lines, width from the lines with the minimum and maximum wavelengths.  In units of wavelength (angstroms)

    :param trim_sigma: [Upper bound, lower bound].  For use in attempting to remove additional lines when identifying
    the continuum (retrim = True).  Keep only what is within [Upper bound, lower bound] sigma from the mean as the
    continuum.  An appropriate choice of trim_sigma will depend on the signal-to-noise of the spectrum

    :param retrim: (True/False) Attempt to trim additional lines from the continuum ?
    :return:
    """

    # ----------------  Check if we are fitting one line or many
    # True if many lines (list, tuple, np.ndarray), False if single line (float)
    are_lines = isinstance(linewaves, (list, tuple, np.ndarray))

    # ---------------- Define a window around the line(s). Large enough to have continuum, small enough to minimize
    # effects of other lines
    if are_lines:
        # If there are many  lines given, take the min and max wavelength of those lines
        short = [np.min(linewaves) - wave_range, np.max(linewaves) + wave_range]
    else:
        # If only one line is given
        short = [linewaves - wave_range, linewaves + wave_range]

    # ---------------- Check the bounds and use the limits of the observed spectrum if needed
    if short[0] < obswave.min():
        short[0] = obswave.min()
    if short[1] > obswave.max():
        short[1] = obswave.max()

    # ---------------- Trim the observed spectrum to the window around the line(s)
    good = np.where((obswave >= short[0]) & (obswave <= short[1]))[0]
    short_wave = obswave[good]
    short_flux = obsflux[good]

    # ---------------- Define a short region of the spectrum +/- line_width from the line center
    if are_lines:
        specline = [ref_wave - line_width, ref_wave + line_width]
    else:
        specline = [linewaves - line_width, linewaves + line_width]

    # ---------------- Check the bounds and use the limits of the trimmed spectrum if needed
    if specline[0] < short_wave.min():
        specline[0] = short_wave.min()
    if specline[1] > short_wave.max():
        specline[1] = short_wave.max()

    # ---------------- Trim the line from the spectrum to find the continuum
    not_line = np.where((short_wave <= specline[0]) | (short_wave >= specline[1]))[0]
    cont_wave = short_wave[not_line]
    cont_flux = short_flux[not_line]

    # ---------------- Attempt to remove other lines
    if retrim:
        cont_flux_mean = np.nanmean(cont_flux)
        cont_flux_std = np.nanstd(cont_flux)
        upper_bound = cont_flux_mean + (trim_sigma[0] * cont_flux_std)
        lower_bound = cont_flux_mean - (trim_sigma[1] * cont_flux_std)
        good = np.where((cont_flux <= upper_bound) & (cont_flux >= lower_bound))[0]

        cont_flux = cont_flux[good]
        cont_wave = cont_wave[good]

    # ---------------- Return the continuum wavelengths and fluxes
    return cont_wave, cont_flux


# -----------------------------------------------------------------------------------------------------------------------
def find_continuum2(wave, flux, sigma_lower, sigma_upper):
    """
    Find wavelength and flux value of the spectrum which constitute the continuum (written by J. Glover)
    --------------
    A function to identify the wavelengths and fluxes of the continuum.
    This is a newer version of the code which uses asymmetric sigma clipping

    Best For:  Spectra which have already been continuum normalized such that they are flat.

    :param wave: (array) An array of wavelength values
    :param flux: (array) An array of corresponding flux values
    :param sigma_lower: (float) The lower sigma value
    :param sigma_upper: (float) The upper sigma value

    :return: clipped wavelength and flux arrays
    (Points with flux outside the specified sigma limits have been removed)
    """

    sol = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=None)

    clip_flux = [flux[i] for i in range(len(sol.mask)) if sol.mask[i] == False]
    clip_wave = [wave[i] for i in range(len(sol.mask)) if sol.mask[i] == False]

    # ---------------- Return the continuum wavelengths and fluxes
    return clip_wave, clip_flux


# -----------------------------------------------------------------------------------------------------------------------
def spectrum_replaceNaN(f, e=None, fill_value=0, change_err=True, fill_error=1e99):
    """
    Replace inf or nan values of flux with a specified value

    :param f: (array) An array of  flux values
    :param e: (array) An array of corresponding error values
    :param fill_value: Value to change all inf and nan to
    :param change_err: (True/False) change the error of points with inf or nan values
    :param fill_error: Value to change the error to

    return The changed flux and error arrays
    """

    mask1 = np.array([np.isnan(val) for val in f])
    mask2 = np.array([np.isinf(val) for val in f])
    mask = np.logical_or(mask1, mask2)

    ind = np.where(mask == True)[0]

    f_change = f.copy()
    f_change[ind] = fill_value

    if e is not None:
        e_change = e.copy()
        if change_err:
            e_change[ind] = fill_error
            return f_change, e_change
    else:
        return f_change
