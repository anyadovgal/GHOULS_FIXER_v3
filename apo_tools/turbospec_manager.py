import os
import subprocess
from abund_utils import atomic_sym_to_num, solar_abund, ALPHAS

WAVE_KEYS = ['lambda_range', 'delta_lambda'],
ABUND_KEYS = ['metals', 'alphas', 'helium', 'rprocess', 'sprocess',
              'elements']
BABSMA_KEYS = ['model', 'vmicro', 'marcs_file_flag']
BSYN_KEYS = ['sph_flag', 'linelists', 'isotopes']


class TurbospecManager:
    '''
    A class to help format the inputs and run Turbospectrum to create synthetic
    spectra.
    '''

    def __init__(self, inpath='.', outpath=None, turbopath='.',
                 turbo_exec='exec-v19.1', auto=False, **kwargs):
        '''
        On initilization input the input and output paths for the data (i.e.,
            the models and output syntheses respectively).

        The "auto" flag allows the user to either manually set the
            Turbospectrum inputs and run each stage (False) or to automatically
            run Turbospectrum with the input kwarg settings (note that users
            should read through the below functions to see what settings can
            be set).

        inpath : string, the path to the input models
        outpath : string, the path to place output synthetic spectra
        turbopath : string, the path to Turbospectrum
        turbo_exec : string, folder location of Turbospectrum executables
        auto: boolean, True will run Turbospectrum automatically, False allows
            for manual setting and running
        '''

        self.inpath = inpath
        if outpath is None:
            self.outpath = inpath
        else:
            self.outpath = outpath
        self.turbopath = turbopath
        self.turbo_exec = f'{turbopath}/{turbo_exec}'
        self._wave_flag = False
        self._abund_flag = False
        self._babsma_flag = False
        self._syn_flag = False

        if auto:
            wave_kwargs = {key: value for key,
                           value in kwargs.items() if key in WAVE_KEYS}
            abund_kwargs = {key: value for key,
                            value in kwargs.items() if key in ABUND_KEYS}
            babsma_kwargs = {key: value for key,
                             value in kwargs.items() if key in BABSMA_KEYS}
            bsyn_kwargs = {key: value for key,
                           value in kwargs.items() if key in BSYN_KEYS}

            self.set_wave(**wave_kwargs)
            self.set_abund(**abund_kwargs)
            self.run_babsma(**babsma_kwargs)
            self.run_bsyn(**bsyn_kwargs)

    def set_wave(self, lambda_range=None, delta_lambda=0.01):
        '''
        This method sets the wavelength range to min - delta_lambda to
            max + delta_lambda to include the end points of the input
            wavelength range.  NOTE:  You can only synthesize 1,000,000
            points at once

        delta_lambda : float, wavelength step size in angstroms,
            defaults to 0.01 angstroms
        lambda_range : len 2 list of floats, [min, max], minimum and maximum
            wavelength of the range you want to synthesize

        '''

        self.delta_lambda = delta_lambda
        self.lambda_min = lambda_range[0] - delta_lambda
        self.lambda_max = lambda_range[1] + delta_lambda
        self._wave_flag = True

    def set_abund(self, metals=0.0, alphas=0.0, helium=0.0, rprocess=0.0,
                  sprocess=0.0, abundances=None, solar_reference='Asplund2009',
                  exclude=['H', 'He']):
        '''
        This method will take the input abundance tags and compute and format
            individual abundances for input into Turbospectrum.  If individual
            abundances are provided they will be used, otherwise elements will
            be alpha-scaled (for the appropriate elements) or solar-scaled to
            the input metallicity.

        Currently helium, and r- and s-process scaling has not been implemented
            and these elements should be entered directly if solar-scaled
            values are not desired.

        Custom solar references can also be input which must be in a dictionary
            format of {'X' : value}.

        metals : float, the metallicity of the synthesis
        alphas : float, the alpha abundance of the synthesis
        helium : float, the helium abundance of the synthesis
            [Currently not implemented]
        rprocess : float, the r-process abundance scaling of the synthesis
            [Currently not implemented]
        sprocess : float, the s-process abundance scaling of the synthesis
            [Currently not implemented]
        abundances : dictionary, of element symbol and log epsilon
            abundance pairs, e.g., {'C' : 8.66,}.
        solar_reference : string or dictionary, users can select from some of
            the default solar_references that are in abund_utils or they can
            input a dictionary of element symbol and log epsilon solar
            abundance pairs e.g., {'C' : 8.66,}.
        exclude : list, elements to not input their abundances
            in the synthesis
        '''
        self.metals = metals
        self.alphas = alphas
        self.helium = helium
        self.rprocess = rprocess
        self.sprocess = sprocess

        if isinstance(solar_reference, str):
            solar_abu = solar_abund(reference=solar_reference)
        elif isinstance(solar_reference, dict):
            solar_abu = solar_reference

        abundance_dict = {elem: abund + metals for elem,
                          abund in solar_abu.items() if elem not in ['H', 'He']}

        for elem in ALPHAS:
            if elem in abundance_dict:
                abundance_dict[elem] += alphas

        if abundances is not None:
            for elem, abund in abundances.items():
                abundance_dict[elem] = abund

        self.abundances = {atomic_sym_to_num(elem): abund for elem,
                           abund in abundance_dict.items() if elem not in exclude}

        self._abund_flag = True

    def run_babsma(self, model=None, vmicro=2.0, marcs_file_flag=False,
                   verbose=False):
        '''
        Runs babsma to generate the opacity files needed for making synthetic
            spectra. This should be run before bsyn.  Wavelengths and
            abundances should be set before running this

        model : string, the file containing the input model atmosphere
        vmicro : float, the microturbulent velocity to use for the synthetic
            spectrum
        marcs_file_flag :  boolean, tells Turbospectrum whether the model is a
            MARCS model atmosphere or not (NOTE:  interpolated MARCS model
            atmospheres should have this flag set to False)
        verbose : boolean, set to True to see the log from babsma

        returns:
        opac_filename : string, the opacity filename and path which can be used
            as an input for running bsyn
        '''
        if not self._wave_flag:
            print('Please set the wavelength range first.')
        if not self._abund_flag:
            print('Abundances were not set.')

        if not os.path.isfile(f'{self.inpath}/{model}'):
            print(f'The model "{self.inpath}/{model}" does not exist')
        else:
            modelpath = f'{self.inpath}/{model}'

        if marcs_file_flag:
            marcs_string = '.true.'
        else:
            marcs_string = '.false.'

        self.model = model
        self.opac_filename = f'{self.inpath}/{model}_opac'

        babsma_dict = {
            'modelpath': modelpath,
            'marcs_string': marcs_string,
            'vmicro': vmicro
        }

        eof_list = self._write_parameters('babsma', **babsma_dict)

        for line in eof_list:
            print(line, end='')
        if verbose:
            stdout = None
            stderr = None
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        os.symlink(f'{self.turbopath}/DATA', 'DATA')

        p = subprocess.Popen([f'{self.turbo_exec}/babsma_lu'],
                             stdin=subprocess.PIPE,
                             stdout=stdout,
                             stderr=stderr)
        for line in eof_list:
            p.stdin.write(line.encode('utf-8'))
        stdout, stderr = p.communicate()

        os.remove('DATA')
        self._babsma_flag = True

        return self.opac_filename

    def run_bsyn(self, opac_filename=None, sph_flag=True, linelists=None,
                 isotopes={}, result_filename=None, verbose=False):
        '''
        Runs bsyn to generate synthetic spectra.  Will write a resulting file
            that gives the wavelengths, normalized and unnormalized fluxes
            for the synthetic spectrum.

        opac_filename : string, filename and path to the desired opacity model
            file, which can be created directly from run_babsma
        sph_flag : boolean, True for spherical radiative transfer and False for
            plane parallel radiative transfer
        linelists : list, gives the names (paths) to the input linelists
        isotopes : dictionary of float : float, each key, value pair should
            have the elemental number followed by the mass as the decimal,
            e.g., {6.012 : 0.9375, 6.013 : 0.625} for carbon 12 and 13
        result_filename : string, optional way to set the filename for the
            output filename
        verbose : boolean, set to True to see the log from bsyn

        returns:
        synthpath : string, filepath to the resulting synthetic spectrum

        '''

        if opac_filename is not None:
            self.opac_filename = opac_filename
        elif not self._babsma_flag:
            print('Please run babsma first to produce and opacity file,'
                  'otherwise enter in an opacity file.')

        if result_filename is None:
            result_filename = f'{self.model}_{self.lambda_min}_{self.lambda_max}.spec'

        synthpath = f'{self.outpath}/{result_filename}'

        if sph_flag:
            sph_flag_string = 'T'
        else:
            sph_flag_string = 'F'

        bsyn_dict = {
            'synthpath': synthpath,
            'sph_flag': sph_flag_string,
            'isotopes': isotopes,
            'linelists': linelists
        }

        eof_list = self._write_parameters('bsyn', **bsyn_dict)

        for line in eof_list:
            print(line, end='')
        if verbose:
            stdout = None
            stderr = None
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        os.symlink(f'{self.turbopath}/DATA', 'DATA')

        p = subprocess.Popen([f'{self.turbo_exec}/bsyn_lu'],
                             stdin=subprocess.PIPE,
                             stdout=stdout,
                             stderr=stderr)
        for line in eof_list:
            p.stdin.write(line.encode('utf-8'))
        stdout, stderr = p.communicate()

        os.remove('DATA')
        self._bsyn_flag = True

        return result_filename

    def _write_parameters(self, version, **kwargs):
        '''
        Creates a list of lines to feed into babsma and bsyn.

        version : string, either babsma or bsyn
        kwargs : dictionary of necessary inputs for babsma or bsyn
        '''

        eof_list = []

        eof_list += [f"'LAMBDA_MIN:'  '{self.lambda_min:.3f}'\n"]
        eof_list += [f"'LAMBDA_MAX:'  '{self.lambda_max:.3f}'\n"]
        eof_list += [f"'LAMBDA_STEP:' '{self.delta_lambda}'\n"]
        if version == 'babsma':
            eof_list += [f"'MODELINPUT:' '{kwargs['modelpath']}'\n"]
            eof_list += [f"'MARCS-FILE:' '{kwargs['marcs_string']}'\n"]
        if version == 'bsyn':
            eof_list += ["'INTENSITY/FLUX:' 'Flux'\n"]
            eof_list += ["'COS(THETA)    :' '1.00'\n"]
            eof_list += ["'ABFIND        :' '.false.'\n"]
        eof_list += [f"'MODELOPAC:' '{self.opac_filename}'\n"]
        if version == 'bsyn':
            eof_list += [f"'RESULTFILE :' '{kwargs['synthpath']}'\n"]
        eof_list += [f"'METALLICITY:'    '{self.metals:.3f}'\n"]
        eof_list += [f"'ALPHA/Fe   :'    '{self.alphas:.3f}'\n"]
        eof_list += [f"'HELIUM     :'    '{self.helium:.3f}'\n"]
        eof_list += [f"'R-PROCESS  :'    '{self.rprocess:.3f}'\n"]
        eof_list += [f"'S-PROCESS  :'    '{self.sprocess:.3f}'\n"]
        eof_list += [f"'INDIVIDUAL ABUNDANCES:'  '{len(self.abundances)}'\n"]
        for elem, abund in self.abundances.items():
            eof_list += [f"{elem}  {abund:.3f}\n"]
        if version == 'bsyn':
            eof_list += [f"'ISOTOPES:'  '{len(kwargs['isotopes'])}'\n"]
            for isotope, fraction in kwargs['isotopes'].items():
                eof_list += [f"{isotope}  {fraction:.6f}\n"]
        if version == 'babsma':
            eof_list += ["'XIFIX:' 'T'\n"]
            eof_list += [f"{kwargs['vmicro']:.3f}\n"]
        if version == 'bsyn':
            eof_list += [f"'NFILES:'  '{len(kwargs['linelists'])}'\n"]
            for linelist in kwargs['linelists']:
                eof_list += [f"{linelist}\n"]
            eof_list += [f"'SPHERICAL:'  '{kwargs['sph_flag']}'\n"]
            eof_list += ["  30\n"]
            eof_list += ["  300.00\n"]
            eof_list += ["  15\n"]
            eof_list += ["  1.30\n"]

        return eof_list
