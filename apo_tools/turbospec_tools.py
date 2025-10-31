import os
import subprocess
import numpy as np
import sys
import sys
from apo_tools.spec_tools import Spectrum

class SpectrumTS(Spectrum):
    def __init__(self, filepath, normed=True):
        if normed:
            usecols = (0, 1)
        else:
            usecols = (0, 2)

        data = np.genfromtxt(filepath, dtype=None, encoding=None,
                             usecols=usecols)
        wave = data[:, 0]
        flux = data[:, 1]
        super().__init__(wave, flux)


class ConvolutionManager:
    '''
    This class runs faltbon which convolves Turbospectrum
        synthetic spectra with a variety of profiles.  The profiles are chosen
        by an integer with the following options:

        1 = Exponential
        2 = Gaussian
        3 = Radial-Tangential (for macroturbulent velocity broadening; Gray
            1978, Solar Phys. 59, 193)
        4 = Rotational (for vsini)

        Users may provide either a FWHM of the broadening profile or the
        velocity which will be converted into the correct values by faltbon.
    '''

    def __init__(self, inpath='.', outpath=None, faltbon_path='.'):
        '''
        Initialization sets up the input, output and faltbon filepaths.

        Inputs

        inpath : string, input path
        outpath : string, output path
        faltbon_path : string, path to faltbon

        '''

        self.inpath = inpath
        self.faltbon_path = faltbon_path

        if outpath is None:
            self.outpath = inpath
        else:
            self.outpath = outpath

    def run_faltbon(self, filename, profile=None, fwhm=None, vel=None,
                    result=None, verbose=False):
        '''
        Takes an input file and runs faltbon with the given profile and
            broadening (input either as a FWHM or velocity -- if both are
            given the FWHM will be used).  If no output filename is provided
            the detaulf will concatenate the profilem broadening and '.convol'

        Inputs

        filename : string, points to the turbospectrum synthetic spectrum to
            convolve
        profile : integer (1-4), indicates what broadening profile to use
            (1=exp, 2=gauss, 3=rad-tan, 4=rot)
        fwhm : float, convolutional broadening FWHM
        vel : float, convolutional broadening velocity
        result : string, output filename

        Output

        result : the resulting filename
        '''

        if profile is None:
            raise TypeError('You must enter a value for profile'
                            '(1=exp, 2=gauss, 3=rad-tan, 4=rot)')

        if fwhm is not None:
            broadening = fwhm
        elif vel is not None:
            broadening = -1 * vel
        else:
            raise TypeError('You must enter a value for either fwhm or vel')

        if result is None:
            result = f'{filename}_{profile}_{broadening}.convol'

        infilepath = f'{self.inpath}/{filename}'
        outfilepath = f'{self.outpath}/{result}'

        eof_list = self._write_parameters(infilepath, outfilepath, profile,
                                          broadening)

        for line in eof_list:
            print(line, end='')
        if verbose:
            stdout = None
            stderr = None
        else:
            stdout = open('/dev/null', 'w')
            stderr = subprocess.STDOUT

        p = subprocess.Popen([f'{self.faltbon_path}/faltbon'],
                             stdin=subprocess.PIPE,
                             stdout=stdout,
                             stderr=stderr)

        for line in eof_list:
            p.stdin.write(line.encode('utf-8'))
        stdout, stderr = p.communicate()

        return result

    def _write_parameters(self, infilepath, outfilepath, profile, broadening):
        '''
        Writes the necessary input for faltbon into a list of lines
        '''

        eof_list = []
        eof_list += [f"{infilepath}\n"]
        eof_list += [f"{outfilepath}\n"]
        eof_list += [f"{broadening:.3f}\n"]
        eof_list += [f"{profile}\n"]

        return eof_list
