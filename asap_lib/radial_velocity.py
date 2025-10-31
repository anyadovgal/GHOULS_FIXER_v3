import os
import sys
from scipy.interpolate import interp1d
import scipy.signal as spsi
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy import interpolate
import scipy.stats as spst
import copy
from scipy.ndimage import median_filter, gaussian_filter1d, gaussian_filter

# -------- Import the other files of functions
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from asap_lib.constants import *
from asap_lib.handleSpectra import *

"""
Radial Velocities

- Code / functions to do radial velocities 
"""

# -----------------------------------------------------------------------------------------------------------------------
rv_elemDict = {
    # --------- Magnesium b lines: 5167A, 5172A, 5183A
    'Mg': {'wmin': 5100, 'wmax': 5250, 'w': [5167, 5172, 5183]},

    # --------- Sodium D lines: 5896A, 5890A
    'Na': {'wmin': 5850, 'wmax': 5940, 'w': [5896, 5890]},

    # --------- Calcium Triplet: 8498A, 8542A and 8662A
    'CaT': {'wmin': 8450, 'wmax': 8700, 'w': [8498, 8542, 8662]},

    # --------- H-alpha: 6565A
    'Ha': {'wmin': 6520, 'wmax': 6600, 'w': [6565]},

    # ---------  H-beta: 4863A
    'Hb': {'wmin': 4820., 'wmax': 4900., 'w': [4863]},

    # --------- Magnesium (telluric)
    'Mg_tell': {'wmin': 4420., 'wmax': 4500., 'w': []},

    # --------- Oxygen (telluric)
    'O_tell': {'wmin': 7750., 'wmax': 7800., 'w': []},

    # --------- H-epsilon 3970.072
    'Hepsilon': {'wmin': 3925., 'wmax': 4015., 'w': [3970.072]},

    # --------- H-delta 4101.734
    'Hdelta': {'wmin': 4056., 'wmax': 4246., 'w': [4101.734]},

    # --------- Sr-II: 4215.524
    'SrII_4215.524': {'wmin': 4080., 'wmax': 4170., 'w': [4215.524]},

    # --------- H-gamma: 4340.462
    'Hgamma': {'wmin': 4295., 'wmax': 4385., 'w': [4340.462]},

    # --------- Ba-II: 4554.030
    'BaII_4554.030': {'wmin': 4509., 'wmax': 4599., 'w': [4554.030]},

    # --------- Fe-II: 4923.922
    'FeII_4923.922': {'wmin': 4879., 'wmax': 4969., 'w': [4923.922]},

    # --------- Fe-I: 4957.596
    'FeI_4957.596': {'wmin': 4912., 'wmax': 5002., 'w': [4957.596]},

    # --------- Fe-II: 5018.435
    'FeII_5018.435': {'wmin': 4973., 'wmax': 5063., 'w': [5018.435]},

    # --------- Fe-I: 5269.537
    'FeI_5269.537': {'wmin': 5224., 'wmax': 5314., 'w': [5269.537]},

    # --------- Ca-I: 6122.217
    'CaI_6122.217': {'wmin': 6077, 'wmax': 6167., 'w': [6122.217]},
}


# -----------------------------------------------------------------------------------------------------------------------
def print_rvelems():
    """
    Print the avaible rv elements
    :return:
    """

    print('Lines which can be used for rv analysis:\n')
    print('{:20}{:<20}{:<20} rest wavelength(s)'.format('name', 'min wavelength', 'max wavelength'))
    print('----------------------------------------------------------------------------------')
    for item in rv_elemDict.keys():
        print('{:20}{:<20}{:<20} {}'.format(item, rv_elemDict[item]['wmin'], rv_elemDict[item]['wmax'],
                                            rv_elemDict[item]['w']))


# -----------------------------------------------------------------------------------------------------------------------
def elemsInRange(wave_spec, wave_temp):
    """
    Find the rv elements that are within the range of the given spectrum and template spectrum
    These are the lines you can use to get radial velocities

    :return: list of lines you can use to get radial velocities
    """

    minWaves = [rv_elemDict[element]['wmin'] for element in rv_elemDict]
    maxWaves = [rv_elemDict[element]['wmax'] for element in rv_elemDict]
    allElems = np.array([element for element in rv_elemDict])

    wmin_spec = np.ceil(min(wave_spec))
    wmax_spec = np.ceil(max(wave_spec))
    ind_spec = np.where((minWaves >= wmin_spec) & (maxWaves <= wmax_spec))[0]

    wmin_temp = np.ceil(min(wave_temp))
    wmax_temp = np.floor(max(wave_temp))
    ind_temp = np.where((minWaves >= wmin_temp) & (maxWaves <= wmax_temp))[0]

    ind = list(set(ind_spec).intersection(ind_temp))

    return allElems[ind]


# -----------------------------------------------------------------------------------------------------------------------
def spec_chop(wave, flux, elems_list):
    """
    Chop a spectrum into sections around specific spectral lines

    wave: the wavelength array of the spectrum to chop
    flux: the corresponding flux array of the spectrum to chop
    elems: a list of elements around whose spectral lines we want to chop
    (must be some combination of ['Mg', 'CaT', 'Ha', 'Hb', 'Hepsilon', Hgamma', 'Hdelta', 'SrII_4215.524',
    'BaII_4554.030', 'FeII_4923.922', 'FeI_4957.596', 'FeII_5018.435', 'FeI_5269.537', 'CaI_6122.217' ] )
    """

    waves = []
    fluxes = []

    for e in elems_list:
        good = np.where((wave >= rv_elemDict[e]['wmin']) & (wave <= rv_elemDict[e]['wmax']))[0]
        waves.append(wave[good])
        fluxes.append(flux[good])

    return waves, fluxes


# -----------------------------------------------------------------------------------------------------------------------
def pyfxcor(wave, flux, s_wave, s_flux, v_tol=5.0, print_vel=False, plot_shift=False, return_corr_wave=False):
    """
    Find the radial velocity shift between the observed and synthetic spectra.
    Restrict the observed and synthetic spectra to regions around certain spectral features

    Inputs
    ------
    waves: An array of wavelengths for the observed spectrum
    fluxes: The fluxes corresponding to the wavelengths in wave

    s_waves: An array of wavelengths for the synthetic / reference spectrum
    s_fluxes: The fluxes corresponding to the wavelengths in s_waves

    plot_shift: If True plot the shifted spectrum for each window (chopped section of spectrum)
    """

    # ----------------- Make copies of the arrays to not overwrite things
    t_w = np.copy(wave)  # Temporary observed wavelength array
    t_f = np.copy(flux)  # Temporary observed flux arrays
    t_s_w = np.copy(s_wave)  # Temporary synthetic wavelength array
    t_s_f = np.copy(s_flux)  # Temporary synthetic  flux arrays

    # ----------------- Interpolate the synthetic spectra on to the observed spectra grid
    '''
    For the synthetic spectrum find a function (f) which gives synthetic flux as a function of synthetic wavelength 
    '''
    f = interpolate.interp1d(t_s_w, t_s_f, kind='cubic', bounds_error=False, fill_value=1.0)
    '''Evaluate this function at the wavelengths of the observed wavelength array. This generates an array of 
    synthetic fluxes corresponding to the same wavelength grid as the observed fluxes (new_tsf) '''
    new_tsf = f(t_w)

    # -----------------
    obs = np.copy(t_f)
    synth = np.copy(new_tsf)

    # ----------------- Regularize the datasets by subtracting off the mean and dividing by the standard deviation
    obs = obs - obs.mean()
    obs = obs / obs.std()

    synth = synth - synth.mean()
    synth = synth / synth.std()

    # ----------------- Find the cross-correlation in pixel values
    '''The cross-correlation works by shifting one array relative to the other and calculating the correlation (
    similarity) between the shifted arrays 

    First, find the cross-correlation between the  observed spectrum and the synthetic spectrum.  xcorr is an array 
    of correlation values, each corresponding to a different pixel shift between the spectra '''
    xcorr = spsi.correlate(obs, synth, method='fft')

    # Create an array (dp) which contains the pixel shifts corresponding to each correlation value in xcorr
    nsamples = obs.size
    dp = np.arange(1 - nsamples, nsamples)

    '''We want the pixel shift corresponding to the highest correlation value - xcorr.argmax() gives the index of the 
    xcorr array corresponding to the highest correlation value - dp[xcorr.argmax()] gives the pixel shift 
    corresponding to the highest correlation value.  We want to select the inverse of this shift to correct for the 
    radial velocity. '''
    pix_shift = -dp[xcorr.argmax()]

    # ----------------- Convert the pixel shift to a velocity
    # Calculate the conversion between pixels and velocity

    # Find the average change in wavelength per pixel
    dispersions = []
    for i in range(len(t_w) - 1):
        dispersions.append(t_w[i + 1] - t_w[i])
    dispersion = np.mean(dispersions)

    # Find the change in wavelength corresponding to the pixel shift
    d_lam = dispersion * pix_shift

    # Let the rest wavelength be the mean of the synthetic wavelength array
    lam = np.median(t_s_w)

    # Get the velocity correction
    vel = s_o_l * (d_lam / lam)

    if print_vel:
        print(vel)

    # ----------------- RV correct the wavelength array
    # If the corrected velocity is too large, assume an error occurred and make no change
    if np.abs(vel) > v_tol:
        print('corrected velocity too large')
        corr_wave = np.copy(t_w)
    # If the radial velocity is small, do the shift
    else:
        corr_wave = np.copy(t_w) * (1.0 + (vel / s_o_l))

    # ----------------- Plot the shifted spectra
    if plot_shift:
        shifted_f = np.roll(t_f, pix_shift)
        plt.figure(figsize=(8, 5))
        plt.plot(t_w, new_tsf, color='tab:orange', linewidth=0.5, label='Template')
        plt.plot(t_w, t_f, color='k', linestyle=':', linewidth=0.5, label='Original Obs')
        plt.plot(t_w, shifted_f, linewidth=0.5, label='Corrected Obs: Shifted flux')
        plt.plot(corr_wave, t_f, linewidth=0.5, label='Corrected Obs: Corrected Wavelengths')
        plt.legend()
        plt.title('plot_shift=True\nShifted Spectrum')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Normalized Flux')
        plt.show()

    # ----------------- Return
    if return_corr_wave:
        return corr_wave, vel
    else:
        return vel


# -----------------------------------------------------------------------------------------------------------------------
def doppler_corr(waves, fluxes, s_waves, s_fluxes, plot_shift=False):
    """
    Apply a Doppler correction to the given spectrum

    Inputs
    -------
    waves: An array of wavelengths for the observed spectrum
    fluxes: The fluxes corresponding to the wavelengths in wave

    s_waves: An array of wavelengths for the synthetic / reference spectrum
    s_fluxes: The fluxes corresponding to the wavelengths in s_waves

    plot_shift: If True plot the shifted spectrum for each window (chopped section of spectrum)
    """

    # ----------------- For each chunk of spectrum...
    vels = []

    for i in range(len(waves)):
        # Give the chunck of observed spectrum and synthetic spectrum to pyfxcor to compute the radial velocity
        # Set the v_tol very high to accept any radial velocity as a solution
        vel = pyfxcor(waves[i], fluxes[i], s_waves[i], s_fluxes[i], v_tol=1e99, plot_shift=plot_shift)
        vels.append(vel)

    # Calculate the difference between the velocities
    v_min = np.min(vels)
    v_max = np.max(vels)

    delta_v = v_max - v_min

    # If delta v is too large...
    if delta_v > 5.0:
        print('Oh shit! Your velocities differ by >5km/s!')
        print('Measured Velocities: %s' % np.around(np.array(vels), 2))

    return np.array(vels)


# -----------------------------------------------------------------------------------------------------------------------
def rvcor(spectra,
          obs_path,
          synth_spec,
          synth_path,
          names,
          ftype_synth=None,
          rv_elems='all',
          manual_rv=None,
          save_out_spec=True,
          out_path=None,
          plot_synth=False,
          plot_obs=False,
          plot_regions=False,
          plot_shift=False,
          plot_corr=True,
          print_spec_info=False,
          print_info=False,
          save_rv_info=False,
          rv_save_name=None,
          print_corr_spec_info=False,
          save_plot=False,
          save_plot_path=None,
          pause=False):
    """
    Radial Velocity Corrections

    Inputs
    -------

    spectra: A list of spectra to which to apply the radial velocity correction

    obs_path: The path to the folder which contains the spectra

    syth_spec: The name of the template or synthetic spectrum

    template_path: The path to the template or synthetic spectrum

    names: Names of the objects corresponding to the spectra

    rv_elems: Elements to do the radial velocity corrections with
    """

    vels = []

    if out_path is None:
        out_path = obs_path

    if save_plot_path is None:
        save_plot_path = obs_path

    # ----------------- Read in synthetic spectra

    # -------- Automatic Detect File Type
    if ftype_synth is None:
        ftype_synth = synth_spec.split('.')[1]

    s_wave, s_flux, *s_err = read_spec(synth_path + synth_spec, ftype=ftype_synth)

    # ----------------- Plot the synthetic (reference) spectra
    if plot_synth is True:
        plt.figure(figsize=(8, 5))
        plt.plot(s_wave, s_flux, color='tab:orange', linewidth=0.5, label='Synthetic')
        # for i in range(len(s_waves)):
        #     plt.plot(s_waves[i], s_fluxes[i], linewidth=0.5, label=rv_elems[i])
        plt.legend()
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel('Normalized Flux')
        plt.title('plt_synth=True\nSynthetic Template')
        plt.show()
        if save_plot is True:
            plt.savefig(save_plot_path + 'SyntheticTemplate.eps', format='eps')
    # -----------------

    # ----------------- Examine (observed) Spectra
    if type(spectra) == str:
        spectra = [spectra]

    if type(names) == str:
        names = [names]

    best_vs = []
    best_vs_err = []

    # ----------------- For each (observed) spectrum:
    for n, s in enumerate(spectra):
        print('')
        print('Correcting ', names[n])

        # ----------------- Automatic Detect File Type
        ftype = s.split('.')[1]
        if ftype not in ['bin', 'xyz']:
            print('Data type must be .bin or .xyz\nSkipping {},{}'.format(n, s))
            continue

        # ----------------- Read in (observed) spectra
        spectrum = read_spec(os.path.join(obs_path, s), ftype=ftype)
        wave = spectrum[0]
        flux = spectrum[1]
        err = spectrum[2]

        # -----------------
        if rv_elems == 'all':
            rv_elems = elemsInRange(wave, s_wave)

        # ----------------- Chop the synthetic spectrum
        '''Isolate and keep only sections of the synthetic spectrum around the lines

            - s_waves is an array with a certain number of sub arrays (= to the number of elements chosen).  Each 
            sub-array corresponds to the spectrum chopped around an element. '''
        s_waves, s_fluxes = spec_chop(s_wave, s_flux, rv_elems)

        # ----------------- Chop the (observed) spectrum
        '''Isolate and keep only sections of the synthetic spectrum around the Mg b lines (5167A, 5172A, 5183A), 
        the Na D lines (5896Å, 5890Å), and/or the Calcium Triplet (8498A, 8542 A, 8662 A) 

        - waves is an array with a certain number of sub arrays (= to the number of elements chosen).  Each sub-array 
        corresponds to the spectrum chopped around an element. '''
        waves, fluxes = spec_chop(wave, flux, rv_elems)

        # ----------------- Print information about the (observed) spectrum:
        if print_spec_info is True:
            print(names[n])
            for i in range(len(waves)):
                print('Length of %s obs spec: ' % rv_elems[i], len(fluxes[i]))
                print('Length of %s obs wave: ' % rv_elems[i], len(waves[i]))
                print('%s obs start: ' % rv_elems[i], np.min(waves[i]))
                print('%s obs end: ' % rv_elems[i], np.max(waves[i]))
                print('Length of %s synth spec: ' % rv_elems[i], len(s_fluxes[i]))
                print('Length of %s synth wave: ' % rv_elems[i], len(s_waves[i]))
                print('Synth %s start: ' % rv_elems[i], np.min(s_waves[i]))
                print('Synth %s end: ' % rv_elems[i], np.max(s_waves[i]))
                print('---')
        # -----------------

        # ----------------- Plot the (observed) spectrum:
        if plot_obs is True:
            plt.figure(figsize=(8, 5))
            plt.plot(wave, flux, color='k', linewidth=0.5, label=names[n])
            for i in range(len(waves)):
                plt.plot(waves[i], fluxes[i], linewidth=0.5, label=rv_elems[i])
            plt.legend()
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.ylabel('Normalized Flux')
            plt.title('plot_obs=True\n: %s' % names[n])
            plt.show()
            if save_plot is True:
                plt.savefig(save_plot_path + '%s_.eps' % names[n], format='eps')

        # ----------------- Plot the "windows" around the specified elements for the (observed) spectrum:
        if plot_regions is True:
            for i in range(len(waves)):
                plt.figure(figsize=(8, 5))
                plt.plot(waves[i], fluxes[i], color='k', linewidth=0.5, label=names[n])
                plt.plot(s_waves[i], s_fluxes[i], color='tab:orange', linewidth=0.5, label='Template')

                # plt.xlim(rv_elemDict[rv_elems[i]]['wmin'], rv_elemDict[rv_elems[i]]['wmax'])

                plt.legend()
                plt.xlabel(r'Wavelength ($\AA$)')
                plt.ylabel('Normalized Flux')
                plt.title('plot_regions=True\n %s lines: %s' % (rv_elems[i], names[n]))
                plt.show()
                if save_plot is True:
                    plt.savefig(save_plot_path + '%s_%s_lines.eps' % (names[n], rv_elems[i]), format='eps')

        # ----------------- Automatic rv correction
        # ''' If  manual_rv is not selected, then do an automatic rv correction
        # '''
        # # if manual_rv is None:
        vels = doppler_corr(waves, fluxes, s_waves, s_fluxes, plot_shift)

        # Calculate the average velocity end error from each line
        best_v = np.mean(vels)
        best_v_err = np.std(vels)

        # # ----------------- Manual rv correction
        # else:
        #     '''If things are wonky, accept a manual input for the RV correction and overwrite some stuff (to be
        #     expanded). Calculate the average RV from the three regions (error bar????) '''
        #     vels = manual_rv
        #     best_v = vels[3]
        #     best_v_err = float(np.std(vels[:-1]))
        # -----------------

        best_vs.append(best_v)
        best_vs_err.append(best_v_err)

        # ----------------- Print information
        if print_info is True:
            # if manual_rv is None:
            for i in range(len(rv_elems)):
                print('Velocity correction from %s lines: ' % rv_elems[i], vels[i])
            # else:
            #     print('By hand velocity from Mg lines: ', vels[0])
            #     print('By hand velocity from Na lines: ', vels[1])
            #     print('By hand velocity from Ca triplet: ', vels[2])
            #     print('By hand velocity from Ha: ', vels[3])
            #     print('By hand velocity from Hb: ', vels[4])
            #     print('By hand velocity from Mg_Tell: ', vels[5])
            #     print('By hand velocity from O_Tell: ', vels[6])

            print(r'Mean radial velocity: %s $\pm$ %s' % (round(best_v, 2), round(best_v_err, 2)))

        # ----------------- RV correct the wavelength array
        '''
        Apply the average radial velocity shift to the entire original wavelength array 
        '''
        corr_wave = np.copy(wave) * (1.0 + best_v / s_o_l)

        # -----------------  Interpolate the original flux onto the corrected wavelength array
        f = interpolate.interp1d(corr_wave, np.copy(flux), kind='cubic', bounds_error=False, fill_value=1.0)
        ef = interpolate.interp1d(corr_wave, np.copy(err), kind='cubic', bounds_error=False, fill_value=1.0)

        # ----------------- Transpose the corrected flux array back onto the original grid
        corr_flux = f(np.copy(wave))
        corr_err = ef(np.copy(wave))

        # -----------------
        if plot_corr is True:
            plt.figure(figsize=(8, 5))
            plt.plot(s_wave, s_flux, color='tab:orange', linewidth=0.5, label='Template')
            plt.plot(wave, flux, color='k', linestyle='-', linewidth=0.5, label='Original Obs')
            plt.plot(wave, corr_flux, color='tab:green', linestyle=':', linewidth=0.5, label='Corrected Obs')
            plt.plot(wave, err, color='tab:red', linestyle='-', linewidth=0.5, label='Original Err')
            plt.plot(wave, corr_err, color='tab:blue', linestyle=':', linewidth=0.5, label='Corrected Err')
            plt.legend()
            plt.ylim(0, 1.5)
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.ylabel('Normalized Flux')
            plt.title('plot_corr=True\nRV Corrected GRACES: %s' % names[n])
            plt.show()
            if save_plot is True:
                plt.savefig(save_plot_path + '%s_RV_Corr.eps' % names[n], format='eps')

        # -----------------
        if print_corr_spec_info is True:
            dl = []
            for i in range(len(wave) - 1):
                dl.append(wave[i + 1] - wave[i])
            dl_best = np.mean(dl)
            print('Starting wavelength: ', np.min(wave))
            print('Ending wavelength: ', np.max(wave))
            print('Min delta lambda: ', np.min(dl))
            print('Max delta lambda: ', np.max(dl))
            print('Mean delta lambda: ', dl_best)
            print('Length of wavelength array: ', len(wave))
            print('Length of corrected flux array: ', len(corr_flux))

        if save_out_spec is True:
            colheads = ['wave', 'flux', 'err']
            table = Table([wave, corr_flux, corr_err], names=colheads)
            # pickle.dump(table,open(out_path+names[n]+'.bin','wb'))
            print('Saving radial velocity corrected spectrum to ' + out_path + s[:-4] + '.rv.bin')
            pickle.dump(table, open(out_path + s[:-4] + '.rv.bin', 'wb'))

        if pause is True:
            input("Press Enter to continue to the next star...")

    # -----------------
    if save_rv_info is True:
        best_vs = -1 * np.array(best_vs)
        colheads = ['name', 'mean_v', 'mean_v_err'] + [rv_elems[i] for i in range(len(rv_elems))]
        tab = [names, best_vs, best_vs_err] + [[vels[i]] for i in range(len(rv_elems))]

        table = Table(tab, names=colheads)
        # ascii.write(table,'GRACESRVs.txt')
        print('Saving radial velocity to ' + rv_save_name)
        ascii.write(table, rv_save_name, overwrite=True)
        # pickle.dump(table,open(out_path+name[n]+'.bin','wb'))
