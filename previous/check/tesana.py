import numpy as np
import Constants_my as Constants
from pylab import figure, plot, errorbar, hist, axvline, xlim, ylim, loglog, xlabel, ylabel, legend, tight_layout, savefig, grid, step
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import cauchy, norm
from scipy.signal import fftconvolve
from scipy.special import wofz
import warnings
import math

def y2i(pha,a,b,c):
    return (-b+np.sqrt((b**2)-(4*a*(c-pha))))/(2*a)

def y2(x,a,b):
    return a*(x**2)+(b*x)

class TES:
    def __init__(self,t,p,n,lpfc=None,hpfc=None,center=False,max_shift=10,gain=11321.813285457809):
        self.t    = t
        dt        = np.diff(self.t)[0]
        self.df   = (dt * self.t.shape[-1])**-1
        self.p    = p
        self.n    = n
        self.tmpl = []
        self.sn   = []
        self.lpfc = lpfc
        self.hpfc = hpfc
        self.max_shift = max_shift
        self.gain = gain
        ofs     = np.median(self.n)
        self.p  = self.p - ofs
        self.n  = self.n - ofs
        if center:
            r = self.p.shape[-1] / 2 - np.median(abs(self.p - offset(self.p)[:, np.newaxis]).argmax(axis=-1))
            self.p = np.hstack((self.p[...,-r:], self.p[...,:-r]))

        self.offset = offset(self.p)
        self.length = int(self.p.shape[1])
        self.avgns  = [] #V/srHz or A/srHz
        self.x_hz   = [] #Hz
        self.pha_p  = []
        self.pha_n  = []
        self.plc    = []
        self.elc    = []
        self.pi_p   = []
        self.pi_n   = []
        self.ka     = []
        self.kb     = []
        self.p_cl   = []
        self.p_j    = []
        self.avgp = average_pulse(p)

    def Noise(self):        
        self.avgns = np.sqrt(average_noise(self.n/self.gain) / self.df)
        self.x_hz = np.arange(len(self.avgns))*self.df
        self.avgns[0] = 0    # for better plot

    def plotFigure(self):
        #average pulse
        figure()
        plot(self.t, self.avgp,'k-')
        xlabel(r'$Time\ (s)$',fontsize=14)
        ylabel(r'$Average\ pulse\ (V)$',fontsize=14)
        tight_layout()

        #Noise
        figure()
        step(self.x_hz, self.avgns*1e12,'k')
        loglog()
        xlabel(r'$Frequency\ (Hz)$',fontsize=14)
        ylabel(r'$Noise\ (pA/\sqrt{\mathrm{Hz}})$',fontsize=14)
        grid(ls='--',which='both')
        tight_layout()

        #SN
        figure()
        plot(self.x_hz, self.sn,'k-')
        loglog()
        xlabel(r'$Frequency\ (Hz)$',fontsize=14)
        ylabel(r'$S/N$',fontsize=14)
        tight_layout()

        #Template
        figure()
        plot(self.t, self.tmpl,'k-')
        xlabel(r'$Time\ (s)$',fontsize=14)
        ylabel(r'$Template\ (a.u.)$',fontsize=14)
        tight_layout()

        #LinearityCorrection
        figure()
        plot(self.elc[1:], self.plc[1:],marker='o',mfc='None',mec='k',ls='None')
        x = np.arange(0,7001,1)
        plot(x,y2(x,self.a,self.b),'r--')
        xlabel(r'$Energy\ (eV)$',fontsize=14)
        ylabel(r'$PHA\ (a.u.)$',fontsize=14)
        tight_layout()

        #PHA spec
        figure()
        hist(self.pha_p, bins=1024, histtype='step', color='k')
        xlabel(r'$PHA\ (a.u.)$',fontsize=14)
        ylabel(r'$Counts$',fontsize=14)
        tight_layout()

        #PI spec
        figure()
        hist(self.pi_p, bins=int(self.pi_p.max()/1), histtype='step', color='k')
        xlabel(r'$PI\ (eV)$',fontsize=14)
        ylabel(r'$Counts/1eV$',fontsize=14)
        tight_layout()

    def dataReduction(self,thre=0.1,repl=True):
        self.boxcar = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.thre   = thre
        self.trig   = 0.1
        self.p_cl   = []
        self.p_j    = []
        p_mask      = []
        count_t = int(self.p.shape[0])
        print('checking double pulse...')
        for e, i in enumerate(self.p):
            p_b  = np.correlate(-i, self.boxcar, mode='same')
            peaks, _ = find_peaks(p_b, thre)
            if len(peaks) == 0 or len(peaks) > 1:
                p_mask.append(False)
                self.p_j.append(i)
            else:
                tmin = self.trig*self.length*0.80
                tmax = self.trig*self.length*1.20
                if (tmin<peaks[0])&(peaks[0]<tmax):
                    self.p_cl.append(i)
                    p_mask.append(True)
                else:
                    p_mask.append(False)
                    self.p_j.append(i)

            print(f'\r{e+1}/{count_t} {(e+1)/count_t*100:.2f}%',end='',flush=True)

        count_j = int(len(self.p_j))
        count_cl = int(len(self.p_cl))
        #print('\n')
        print(f'\nNumber of Junck pulse  {count_j}')
        print(f'Number of clean events {count_cl}')
        if count_t == 0:
            pass
        else:
            print(f'Pulse removal ratio {count_cl/count_t*100:.2f}%\n')

        self.p_cl = np.asarray(self.p_cl)
        self.p_j = np.asarray(self.p_j)
        if repl:
            self.p = self.p_cl
            del self.p_cl
            self.offset = offset(self.p)

    def PHA(self, mask=None):
        if mask:
            self.p = self.p[mask]
            self.n = self.n[mask]
        self.tmpl, self.sn = generate_template(self.p, self.n, lpfc=self.lpfc, hpfc=self.hpfc, max_shift=self.max_shift)
        print("Resolving power: %.2f (%.2f eV @ 5.9 keV)" % (np.sqrt((self.sn**2).sum()*2), baseline(self.sn)))
        self.pha_p, ps = optimal_filter(self.p, self.tmpl, max_shift=self.max_shift)
        self.pha_n, ns = optimal_filter(self.n, self.tmpl, max_shift=0)
        self.Noise()

    def Offset_correction(self):
        oc_pha_p, para, coef = offset_correction(self.pha_p, self.offset)
        #oc_pha_n, para, coef = offset_correction(self.pha_n, self.offset)
        self.pha_p = oc_pha_p
        #self.pha_n = oc_pha_n

    def SetPHAregion(self, kamin, kamax, kbmin, kbmax):
        self.ka_min_pha = kamin
        self.ka_max_pha = kamax
        self.kb_min_pha = kbmin
        self.kb_max_pha = kbmax
        self.ka_pha = self.pha_p[ (self.ka_min_pha<self.pha_p)&(self.pha_p<self.ka_max_pha) ]
        self.kb_pha = self.pha_p[ (self.kb_min_pha<self.pha_p)&(self.pha_p<self.kb_max_pha) ]
        self.checkregion = True

    def LinearityCorrection(self, sb=5):
        sb = sb
        # nc , bins, patches = hist(self.pha_p,bins=1024,histtype='step',cplor='k')
        # bins = (bins[1:]+bins[:-1])/2
        # pha_s = np.convolve(nc, np.ones(sb)/sb, mode='same')
        # peaks, _ = find_peaks(pha_s,30)
        # ka = bins[nc[peaks].argmax()]
        # kb = bins[nc[peaks].argmin()]
        #self.plc = np.asarray([0, ka, kb])

        ka_pha_center = np.median(self.ka_pha)
        kb_pha_center = np.median(self.kb_pha)

        self.plc = np.asarray([0, ka_pha_center, kb_pha_center])
        self.elc = np.asarray([0, Constants.LE['MnKa'], Constants.LE['MnKb'] ])
        a, b, c = np.polyfit(self.elc, self.plc, deg=2)
        self.a = a
        self.b = b
        self.pi_p = y2i(self.pha_p, a, b, c)
        self.pi_n = y2i(self.pha_n, a, b, c)
        l = (kb_pha_center/ka_pha_center)/(Constants.LE['MnKb']/Constants.LE['MnKa'])
        print(f'{l*100}%')

    def PI(self,ofsc=False):
        self.Noise()
        if ofsc:
            self.Offset_correction()
            
        self.LinearityCorrection()

    def LineFitting(self,method="c",sigma=3,binsize=2,ka_min=10,kb_min=10):
        from scipy.stats import norm
        method  = method
        sigma   = sigma
        binsize = binsize
        ka_min  = ka_min
        kb_min  = ka_min
        atom    = 'Mn'
        plotting = True
        shift = False
        kbfit = True

        def _line_fit(data, min, line):
            # Fit
            (dE, dEc), (dE_error, dEc_error), e = fit(data, binsize=binsize, gmin=min, line=line, method=method)

            if method == "cs":
                chi_squared, dof = e

            if method in ("c", "mle", "ls"):
                print( "%s: %.2f +/- %.2f eV @ Ec%+.2f eV" % (line, dE, dE_error, dEc))
            elif method == "cs":
                print( "%s: %.2f +/- %.2f eV @ Ec%+.2f eV (Red. chi^2 = %.1f/%d = %.2f)" % (line, dE, dE_error, dEc, chi_squared, dof, chi_squared/dof))

            return dEc, dE, dE_error        


        def _line_spectrum(data, min, line, dEc, dE, dE_error, method):

            # Draw histogram
            n, bins = histogram(data, binsize=binsize)

            if method == "cs":
                gn, gbins = group_bin(n, bins, min=min)
            else:
                # No grouping in mle and ls
                gn, gbins = n, bins

            ngn = gn/(np.diff(gbins))
            ngn_sigma = np.sqrt(gn)/(np.diff(gbins))
            cbins = (gbins[1:]+gbins[:-1])/2

            if plotting:
                figure()

                if dE_error is not None:
                    label = 'FWHM$=%.2f\pm %.2f$ eV' % (dE, dE_error)
                else:
                    label = 'FWHM$=%.2f$ eV (Fixed)' % dE

                if method == "cs":
                    errorbar(cbins, ngn, yerr=ngn_sigma, xerr=np.diff(gbins)/2, capsize=0, ecolor='k', fmt='None', label=label)
                else:
                    hist(data, bins=gbins, weights=np.ones(len(data))/binsize, histtype='step', ec='k', label=label)

                E = np.linspace(bins.min(), bins.max(), 1000)

                model = len(data)*line_model(E, dE, dEc, line=line, shift=shift, full=True)

                # Plot theoretical model
                plot(E, model[0], 'r-')

                # Plot fine structures
                for m in model[1:]:
                    plot(E, m, 'b--')

                xlabel('Energy$\quad$(eV)')
                ylabel('Normalized Count$\quad$(count/eV)')
                legend(frameon=False)

                ymin, ymax = ylim()
                ylim(ymin, ymax*1.1)
                tight_layout()


        ## Ka
        self.ka_pi = self.pi_p[self.pi_p==self.pi_p][(Constants.LE['MnKa']-50<self.pi_p)&(self.pi_p<Constants.LE['MnKa']+50)]
        dEc, dE, dE_error = _line_fit(self.ka_pi, ka_min, 'MnKa')
        _line_spectrum(self.ka_pi, ka_min, 'MnKa', dEc, dE, dE_error, method)

        ## Kb
        self.kb_pi = self.pi_p[self.pi_p==self.pi_p][(Constants.LE['MnKb']-50<self.pi_p)&(self.pi_p<Constants.LE['MnKb']+50)]
        if kbfit:
            dEc, dE, dE_error = _line_fit(self.kb_pi, kb_min, 'MnKb')
        else:
            dE_error = None
        _line_spectrum(self.kb_pi, kb_min, 'MnKb', dEc, dE, dE_error, method)

        ## Baseline
        baseline = sigma2fwhm(np.std(self.pi_n))
        print("Baseline resolution: %.2f eV" % baseline)
    
        n, bins = histogram(self.pi_n, binsize=binsize)

        if plotting:
            figure()
            label = 'FWHM$=%.2f$ eV' % baseline
            hist(self.pi_n, bins=bins, weights=np.ones(len(self.pi_n))/binsize, histtype='step', ec='k', label=label)
            mu, sigma = norm.fit(self.pi_n)
            E = np.linspace(bins.min(), bins.max(), 1000)
            plot(E, norm.pdf(E, loc=mu, scale=sigma)*len(self.pi_n), 'r-')
        
            xlabel('Energy$\quad$(eV)')
            ylabel('Normalized Count$\quad$(count/eV)')
        
            legend(frameon=False)
        
            tight_layout()    
  
#from Sakai-san script
def sigma2fwhm(sigma):
    """
    Convert sigma to width (FWHM)
    
    Parameter:
        sigma:  sigma of gaussian / voigt profile
    
    Return (fwhm)
        fwhm:   width
    """
    
    return 2*sigma*np.sqrt(2*np.log(2))

def fwhm2sigma(fwhm):
    """
    Convert width (FWHM) to sigma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        sigma:  sigma of gaussian / voigt profile
    """
    
    return fwhm/(2*np.sqrt(2*np.log(2)))

def fwhm2gamma(fwhm):
    """
    Convert width (FWHM) to gamma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        gamma:  gamma of lorentzian / voigt profile
    """
    
    return fwhm/2.0

def fwhm2gamma(fwhm):
    """
    Convert width (FWHM) to gamma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        gamma:  gamma of lorentzian / voigt profile
    """
    
    return fwhm/2.0

def baseline(sn, E=5.9e3):
    """
    Calculate a baseline resolution dE(FWHM) for the given energy.
    
    Parameter:
        sn:     S/N ratio (array-like)
        E:      energy to calculate dE
    """
    
    return 2*np.sqrt(2*np.log(2))*E/np.sqrt((sn**2).sum(axis=-1)*2)

def offset(pulse, bins=None, avg_pulse=None):
    """
    Calculate an offset (DC level) of pulses
    
    Parameters (and their default values):
        pulse:      pulses (N or NxM array-like)
        bins:       tuple of (start, end) for bins used for calculating an offset
                    (Default: None = automatic determination)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        avg_pulse:  if given, use this for averaged pulse (Default: None)

    Return (offset)
        offset: calculated offset level
    """
    
    pulse = np.asarray(pulse)
    
    if bins is None:
        if avg_pulse is None:
            avg_pulse = average_pulse(pulse)
        i = np.correlate(avg_pulse, [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1]).argmax() - 16
        if i < 1:
            raise ValueError("Pre-trigger is too short")
        return pulse[..., :i].mean(axis=-1)
    else:
        return pulse[..., bins[0]:bins[1]].mean(axis=-1)

def median_filter(arr, sigma):
    """
    Noise filter using Median and Median Absolute Deviation for 1-dimentional array
    """
    
    if sigma is None:
        return np.ones(arr.size, dtype='b1')
    
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
                
    # Tiny cheeting for mad = 0 case
    if mad == 0:
        absl = np.abs(arr - med)
        if len(absl[absl > 0]) > 0:
            mad = (absl[absl > 0])[0]
        else:
            mad = np.std(arr) / 1.4826
                
    return (arr >= med - mad*1.4826*sigma) & (arr <= med + mad*1.4826*sigma)

def reduction(data, sigma=3, **kwargs):
    """
    Do data reduction with sum, max and min for pulse/noise using median filter (or manual min/max)
    
    Parameters (and their default values):
        data:   array of pulse/noise data (NxM or N array-like)
        sigma:  sigmas allowed for median filter
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (mask)
        mask:   boolean array for indexing filtered data
    """
    
    data = np.asarray(data)
    
    if "min" in kwargs:
        min_mask = (data.min(axis=1) > kwargs["min"][0]) & (data.min(axis=1) < kwargs["min"][1])
    else:
        min_mask = median_filter(data.min(axis=1), sigma)

    if "max" in kwargs:
        max_mask = (data.max(axis=1) > kwargs["max"][0]) & (data.max(axis=1) < kwargs["max"][1])
    else:
        max_mask = median_filter(data.max(axis=1), sigma)

    if "sum" in kwargs:
        sum_mask = (data.sum(axis=1) > kwargs["sum"][0]) & (data.sum(axis=1) < kwargs["sum"][1])
    else:
        sum_mask = median_filter(data.sum(axis=1), sigma)

    return min_mask & max_mask & sum_mask


def average_pulse(pulse):

    pulse = np.asarray(pulse)
    
    s = []
    
    # Calculate averaged pulse
    if pulse.ndim == 2:
        avg_pulse = np.average(pulse, axis=0)
    
    elif pulse.ndim == 1:
        # Only one pulse data. No need to average
        avg_pulse = pulse
    
    else:
        raise ValueError("object too deep for desired array")

    return avg_pulse

def power(data):
    
    data = np.asarray(data)
    
    # Real DFT
    ps = np.abs(np.fft.rfft(data) / data.shape[-1])**2
    
    if data.shape[-1] % 2:
        # Odd
        ps[...,1:] *= 2
    else:
        # Even
        ps[...,1:-1] *= 2
    
    return ps

def average_noise(noise, sigma=3, r=0.2, rr=0.1, **kwargs):
    """
    Calculate averaged noise power
    
    Parameters (and their default values):
        noise:      array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        r:          amount in ratio of removal in total for data reduction (Default: 0.2)
        rr:         amount in ratio of removal for each step for data reduction (Default: 0.1)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (averaged_pulse)
        power_noise:    calculated averaged noise power in V^2
    """

    # Convert to numpy array
    noise = np.asarray(noise)

    if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
        noise = noise[reduction(noise, sigma, **kwargs)]
        
        nlen = len(noise)

        while (len(noise) > (nlen*(1.0-r))):
            avg = np.average(power(noise), axis=0)
            noise = noise[((power(noise) - avg)**2).sum(axis=-1).argsort() < (len(noise) - nlen*rr - 1)]

    return np.average(power(noise), axis=0)

def offset_correction(pha, offset, sigma=1, prange=None, orange=None, flip=False, atom='Mn', tail=False, ignorekb=False, p=None, filename=None, tex=False):
    """
    Fit a pha and an offset (DC level)
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        offset:     offset data (array-like)
        sigma:      sigmas allowed for median filter (Default: 1)
        prange:     a tuple of range for pha to fit if not None (Default: None)
        orange:     a tuple of range for offset to fit if not None (Default: None)
        flip:       flip pha and offset when fitting if True (Default: False)
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas for median filter (Default: 1)
        tail:       enable low energy tail (Default: False)
        ignorekb:   do not use Kb (Default: False)
        p:          polynomial coefficients to use for correction (Default: None)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (pha, p, coef):
        pha:    offset corrected pha
        p:      fitting results to pha = offset*p[1] + p[0]
        coef:   correlation coefficient

    Note:
        - If p is given, it just uses it to perform the correction
        - If p is not given, it first makes it and uses it
    """

    # Sanity check
    if len(pha) != len(offset):
        raise ValueError("data length of pha and offset does not match")
    
    pha = np.asarray(pha)
    offset = np.asarray(offset)
    
    if p is None:
        # Reduction
        if prange is not None:
            pmask = (pha >= prange[0]) & (pha <= prange[1])
        else:
            pmask = median_filter(pha, sigma=sigma)

        if orange is not None:
            omask = (offset >= orange[0]) & (offset <= orange[1])
        else:
            omask = median_filter(offset, sigma=sigma)

        mask = pmask & omask

        # Correlation coefficient
        coef = np.corrcoef(pha[mask], offset[mask])[0,1]

        # Pre-fit to a*x+b
        if flip:
            p = np.polyfit(pha[mask], offset[mask], 1)
            p = np.array([p[0]**-1, -p[1]/p[0]])
        else:
            p = np.polyfit(offset[mask], pha[mask], 1)
        
        # Fiting func
        def dE(p0):
            # Fit p1
            popt, pcov = curve_fit(lambda x, p1: x*p0+p1, offset[mask], pha[mask])
            p = [ p0, popt[0] ]
            
            # Perform offset correction
            oc_pha = pha / np.polyval(p, offset)
            
            # Linearity correction
            lc_pha, lc_r, lc_p = linearity_correction(oc_pha, method="fitting", atom=atom, sigma=sigma, tail=tail, ignorekb=ignorekb)
            
            # Correction using K-alpha
            ka_pha = ka(lc_pha, sigma=sigma)
            
            # Line fit and return dE
            return fit(ka_pha, line=atom+"Ka", tail=tail, method="c", error=False)[0][0]
        
        # Minimize
        res = minimize(dE, x0=[p[0]], method='Nelder-Mead')
        
        if not res['success']:
            raise Exception("Fitting failed for %s in offset correction" % line)
        
        p0 = res['x'][0].tolist()

        # Fit p1
        popt, pcov = curve_fit(lambda x, p1: x*p0+p1, offset[mask], pha[mask])
        p = [ p0, popt[0] ]
        
        # Plot if needed
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['text.usetex'] = str(tex)
            from matplotlib import pyplot as plt

            fig = plt.figure()

            ax = plt.subplot(211)
            plt.plot(offset[mask], pha[mask], ',', c='k')
            x_min, x_max = plt.xlim()
            x = np.linspace(x_min, x_max)
            label = '$\mathrm{PHA}=\mathrm{Offset}\\times%.2f+%.2f$' % tuple(p)
            plt.plot(x, np.polyval(p, x), 'r-', label=label)
            plt.xlabel('Offset$\quad$(a.u.)')
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.legend(frameon=False)
            
            ax = plt.subplot(212)
            plt.plot(offset[mask], pha[mask] / np.polyval(p, offset[mask]), ',', c='k')
            plt.axhline(1, color='r', ls='--')
            plt.xlim(x_min, x_max)
            plt.xlabel('Offset$\quad$(a.u.)')
            plt.ylabel('PHA$\quad$(a.u.)')
            
            plt.tight_layout()
            plt.savefig(filename)
    else:
        coef = None
    
    return pha / np.polyval(p, offset), p, coef

def generate_template(pulse, noise, cutoff=None, lpfc=None, hpfc=None, nulldc=False, **kwargs):
    """
    Generate a template of optimal filter

    Parameters (and their default values):
        pulse:  array of pulse data, will be averaged if dimension is 2
        noise:  array of noise data, will be averaged if dimension is 2
        cutoff: low-pass cut-off bin number for pulse spectrum (Default: None)
                (**note** This option is for backward compatibility only. Will be removed.)
        lpfc:   low-pass cut-off bin number for pulse spectrum (Default: None)
        hpfc:   high-pass cut-off bin number for pulse spectrum (Default: None)
        nulldc: nullify dc bin of template (Default: False)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (template)
        template:   generated template
        sn:         calculated signal-to-noise ratio
    """
    
    # Average pulse
    if pulse.ndim == 2:
        avg_pulse = average_pulse(pulse)
    else:
        avg_pulse = pulse

    # Real-DFT
    fourier = np.fft.rfft(avg_pulse)
    
    # Apply low-pass/high-pass filter
    m = len(avg_pulse)
    n = len(fourier)
    
    if lpfc is None and cutoff is not None:
        lpfc = cutoff
    
    if lpfc is not None and 0 < lpfc < n:
        h = np.blackman(m)*np.sinc(np.float(lpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        fourier *= np.abs(np.fft.rfft(h))

    # Apply high-pass filter
    if hpfc is not None and 0 < hpfc < n:
        h = np.blackman(m)*np.sinc(np.float(hpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        fourier *= (1 - np.abs(np.fft.rfft(h)))
    
    # Null DC bin?
    if nulldc:
        fourier[0] = 0
    
    # Calculate averaged noise power
    if noise.ndim == 2:
        pow_noise = average_noise(noise, **kwargs)
    else:
        pow_noise = noise
    
    # Calculate S/N ratio
    sn = np.sqrt(power(np.fft.irfft(fourier, len(avg_pulse)))/pow_noise)
    
    # Generate template (inverse Real-DFT)
    template = np.fft.irfft(fourier / pow_noise, len(avg_pulse))
    
    # Normalize template
    norm = (avg_pulse.max() - avg_pulse.min()) / ((template * avg_pulse).sum() / len(avg_pulse))
    
    return template * norm, sn

def cross_correlate(data1, data2, max_shift=None, method='interp'):

    # Sanity check
    if len(data1) != len(data2):
        raise ValueError("data length does not match")

    # if given data set is not numpy array, convert them
    data1 = np.asarray(data1).astype(dtype='float64')
    data2 = np.asarray(data2).astype(dtype='float64')
    
    # Calculate cross correlation
    if max_shift == 0:
        return np.correlate(data1, data2, 'valid')[0] / len(data1), 0, 0
    
    # Needs shift
    if max_shift is None:
        max_shift = len(data1) // 2
    else:
        # max_shift should be less than half data length
        max_shift = min(max_shift, len(data1) // 2)
    
    # Calculate cross correlation
    cor = np.correlate(data1, np.concatenate((data2[-max_shift:], data2, data2[:max_shift])), 'valid')
    ind = cor.argmax()

    if method == 'interp' and 0 < ind < len(cor) - 1:
        return (cor[ind] - (cor[ind-1] - cor[ind+1])**2 / (8 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))) / len(data1), ind - max_shift, (cor[ind-1] - cor[ind+1]) / (2 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))
    elif method == 'integ':
        return sum(cor), 0, 0
    elif method in ('none', 'interp'):
        # Unable to interpolate, and just return the maximum
        return cor[ind] / len(data1), ind - max_shift, 0
    else:
        raise ValueError("Unsupported method")

def optimal_filter(pulse, template, max_shift=None, method='interp'):
    """
    Perform an optimal filtering for pulse using template
    
    Parameters (and their default values):
        pulse:      pulses (NxM array-like)
        template:   template (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        method:     interp - perform interpolation for obtained pha and find a maximum
                             (only works if max_shift is given)
                    integ  - integrate for obtained pha
                    none   - take the maximum from obtained pha
                    (Default: interp)
    
    Return (pha, lagphase)
        pha:        pha array
        phase:      phase array
    """
    
    return np.apply_along_axis(lambda p: cross_correlate(template, p, max_shift=max_shift, method=method), 1, pulse)[...,(0,2)].T

def group_bin(n, bins, min=100):
    """
    Group PHA bins to have at least given number of minimum counts
    
    Parameters (and their default values):
        n:      counts
        bins:   bin edges
        min:    minimum counts to group (Default: 100)
    
    Return (grouped_n, grouped_bins)
        grouped_n:      grouped counts
        grouped_bins:   grouped bin edges
    """
    
    grp_n = []
    grp_bins = [bins[0]]

    n_sum = 0

    for p in zip(n, bins[1:]):
        n_sum += p[0]
        
        if n_sum >= min:
            grp_n.append(n_sum)
            grp_bins.append(p[1])
            n_sum = 0
    
    return np.asarray(grp_n), np.asarray(grp_bins)

def voigt(E, Ec, nw, gw):
    """
    Voigt profile
     
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural (lorentzian) width (FWHM)
        gw:     gaussian width (FWHM)
    
    Return (voigt)
        voigt:  Voigt profile
    """
    
    # Sanity check
    if gw == 0:
        return lorentzian(E, Ec, nw)
    
    z = (E - Ec + 1j*fwhm2gamma(nw)) / (fwhm2sigma(gw)*np.sqrt(2))

    return wofz(z).real / (fwhm2sigma(gw)*np.sqrt(2*np.pi))

def gaussian(E, Ec, width):
    """
    Gaussian profile
    
    Parameters:
        E:      energy
        Ec:     center energy
        width:  width (FWHM)
    
    Return (gauss)
        gauss:  Gaussian profile
    """

    sigma = fwhm2sigma(width)

    return norm.pdf(E, loc=Ec, scale=sigma)
    
def lorentzian(E, Ec, nw):
    """
    Lorentzian profile
    
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural width (FWHM)
    
    Return (lorentz)
        lorentz:  Lorentzian profile
    """

    gamma = fwhm2gamma(nw)

    return cauchy.pdf(E, loc=Ec, scale=gamma)


def line_model(E, dE=0, dEc=0, tR=None, tT=None, line="MnKa", shift=False, tail=False, full=False):
    """
    Line model

    Parameters (and their default values):
        E:      energy in eV (array-like)
        dE:     FWHM of gaussian profile in eV (Default: 0 eV)
        dEc:    shift from center energy in eV (Default: 0 eV)
        tR:     low energy tail ratio (Default: None)
        tT:     low energy tail tau (Default: None)
        line:   line (Default: MnKa)
        shift:  treat dEc as shift if True instead of scaling (Default: False)
        tail:   enable low energy tail (Default: False)
        full:   switch for return value (Default: False)

    Return (i) when full = False or (i, i1, i2, ...) when full = True
        i:      total intensity
        i#:     component intensities

    Note:
        If shift is False, adjusted center energies ec_i of fine structures
        will be

            ec_i = Ec_i * (1 + dEc/Ec)

        where Ec_i is the theoretical (experimental) center energy of fine
        structures and Ec is the center energy of the overall profile, which
        is the weighted sum of each component profiles.

        If shift is True, ec_i will simply be

            ec_i = Ec_i + dEc.
    """

    # Sanity check
    if line not in Constants.LE:
        raise ValueError("No data for %s" % line)

    if line not in Constants.FS:
        raise ValueError("No data for %s" % line)

    # Boundary check
    dE = 0 if dE < 0 else dE

    # Center energy
    Ec = np.exp(np.log(np.asarray(Constants.FS[line])[:,(0,2)]).sum(axis=1)).sum()

    if shift:
        model = np.array([ p[2] * voigt(E, p[0]+dEc, p[1], dE) for p in Constants.FS[line] ])

        if tail and tR is not None and tT is not None:
            model = model * (1 - tR) + np.array([ p[2] * tR * voigt_with_tail(E, p[0]+dEc, p[1], dE, tT) for p in Constants.FS[line] ])
    else:
        model = np.array([ p[2] * voigt(E, p[0]*(1+(0 if Ec == 0 else dEc/Ec)), p[1], dE) for p in Constants.FS[line] ])

        if tail and tR is not None and tT is not None:
            model = model * (1 - tR) + np.array([ p[2] * tR * voigt_with_tail(E, p[0]*(1+(0 if Ec == 0 else dEc/Ec)), p[1], dE, tT) for p in Constants.FS[line] ])

    if full:
        return np.nan_to_num(np.vstack((model.sum(axis=0)[np.newaxis], model)))
    else:
        return np.nan_to_num(model.sum(axis=0))

def histogram(pha, binsize=1.0):
    """
    Create histogram
    
    Parameter:
        pha:        pha data (array-like)
        binsize:    size of bin in eV (Default: 1.0 eV)
    
    Return (n, bins)
        n:      photon count
        bins:   bin edge array
    
    Note:
        - bin size is 1eV/bin.
    """
    
    # Create histogram
    bins = np.arange(np.floor(pha.min()/binsize)*binsize, np.ceil(pha.max()/binsize)*binsize+binsize, binsize) 
    n, bins = np.histogram(pha, bins=bins)
    
    return n, bins


def fit(pha, binsize=1, gmin=None, line="MnKa", shift=False, tail=False, freeze=None, method='c', error=True, filename=None, tex=False):
    """
    Fitting of line spectrum
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        binsize:    size of energy bin in eV for histogram (Default: 1)
        gmin:       minimum counts to group bins or None for default (Default: None)
        line:       line to fit (Default: MnKa)
        shift:      treat dEc as shift if True instead of scaling (Default: False)
        tail:       enable low energy tail (Default: False)
        freeze:     array-like of (dE, dEc, tR, tT) to freeze or None to thaw (Default: None)
                    (tR and tT are needed only when tail is set to True)
        method:     fitting method among c/mle/cs/ls (Default: c)
        error:      calculate error (Default: True)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (dE, dEc), (dE_error, dEc_error), (stat, dof) when tail is False,
        or (dE, dEc, tR, tT), (dE_error, dEc_error, tR_error, tT_error), (stat, dof) when tail is True

        dE:             fitted gaussian dE (FWHM)
        dEc:            shift from line center
        tR:             fitted low energy tail ratio
        tT:             fitted low energy tail tau
        dE_error:       dEc error (1-sigma)
        dEc_error:      dEcc error (1-sigma)
        tR_error:       tR error (1-sigma)
        tT_error:       tT error (1-sigma)
        stat:           statistics
        dof:            number of degree of freedom
    """
    
    # Sanity check
    if line not in Constants.FS:
        raise ValueError("No data for %s" % line)
    
    # Create histogram
    n, bins = histogram(pha, binsize=binsize)
    
    # Group bins
    if gmin is None:
        if method in ('c', 'mle'):
            gmin = 1
        else:
            gmin = 10

    gn, gbins = group_bin(n, bins, gmin)
    ngn = gn/np.diff(gbins)   # normalized counts in counts/eV
    ngn_sigma = np.sqrt(gn)/np.diff(gbins)
    
    bincenters = (gbins[1:]+gbins[:-1])/2

    def stat(args, bounds=None):
        # arg = (dE, dEc, tR, tT)
        
        # Boundary check
        if bounds is not None:
            for x, (lower, upper) in zip(args, bounds):
                if not (upper >= x >= lower):
                    # Out of boundary
                    return np.inf
        
        # Model
        m = len(pha)*line_model(bincenters, *args, line=line, shift=shift, tail=tail)
        
        # Truncation (only for C stat and MLE)
        mask = m > 1e-25
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            if method == 'c':
                s = 2*(m[mask] - ngn[mask] + ngn[mask]*(np.log(ngn[mask]) - np.log(m[mask]))).sum()
            elif method == 'mle':
                s = (-np.log(m[mask])*ngn[mask]).sum()
            elif method == 'cs':
                s = ((ngn-m)**2/ngn_sigma**2).sum()
            elif method == 'ls':
                s = ((ngn-m)**2).sum()
            else:
                raise ValueError("Unsupported fitting method %s" % method)
        
            return s

    # Build initial params and bounds (x0 will be ignored)
    x0 = (np.std(pha)/2, 0)
    bounds = ((0.01, np.inf), (-np.inf, np.inf))
    initial_simplex = ((0.5, -1), (0.5, 1), (5, -1))
    
    if tail:
        x0 += (0.2, 5)
        bounds += ((0, 1), (0.01, np.inf))
        initial_simplex = (
            (0.5, -1, 0.05, 1),
            (0.5, 1, 0.05, 1),
            (5, -1, 0.05, 1),
            (0.5, -1, 0.5, 1),
            (0.5, -1, 0.05, 10))

    # Parameter freezing
    if freeze is not None:
        # Backup originals
        _x0 = x0
        _bounds = bounds
        _initial_simplex = initial_simplex
        _stat = stat
        
        x0 = [ [_x] if _f is None else [] for _x, _f in zip(_x0, freeze) ]
        bounds = [ [_b] if _f is None else [] for _b, _f in zip(_bounds, freeze) ]
        
        x0 = tuple(reduce(lambda x, y: x+y, x0))
        bounds = tuple(reduce(lambda x, y: x+y, bounds))
        
        mask = [True] + [ True if _f is None else False for _f in freeze ]
        initial_simplex = np.array(initial_simplex)[mask]
        initial_simplex = initial_simplex.T[mask[1:]].T
        
        # Create wrapper function for stat
        def stat(args, bounds=None):
            # Replace args with frozen params
            args = list(args)
            _args = tuple([ args.pop(0) if _x is None else _x for _x in freeze ])
            
            if bounds is not None:
                # Rebuild bounds array
                bounds = list(bounds)
                _bounds = tuple([ bounds.pop(0) if _x is None else (-np.inf, np.inf) for _x in freeze ])
            else:
                _bounds = None
            
            return _stat(_args, _bounds)
    
    if freeze is not None and reduce(lambda x, y: x&y, [ True if _f is not None else False for _f in freeze ]):
        # All parameters are frozen
        x = x0
        s = stat(x0)
        dof = len(bincenters)
    else:
        # Minimize
        res = minimize(stat, x0=x0, method='Nelder-Mead', args=(bounds,), options={'initial_simplex': initial_simplex, 'maxiter': 500*len(x0)})
    
        if not res['success']:
            raise Exception("Fitting failed for %s" % line)

        x = res['x']
        s = res['fun']
        dof = len(bincenters) - len(x)
    
    if freeze is None:
        dE, dEc = x[:2]
        
        if tail:
            tR, tT = x[2:]
        else:
            tR, tT = 0, 0
    else:
        _x = list(x)
        
        dE, dEc = [ _x.pop(0) if _f is None else _f for _f in freeze[:2] ]
        
        if tail:
            tR, tT = [ _x.pop(0) if _f is None else _f for _f in freeze[2:] ]
        else:
            tR, tT = 0, 0

    if freeze is not None and reduce(lambda x, y: x&y, [ True if _f is not None else False for _f in freeze ]):
        # All parameters are frozen
        dE_e, dEc_e, tR_e, tT_e = -1, -1, -1, -1
    else:
        # Calculate Hessian matrix for standard error
        if freeze is None:
            dE_e, dEc_e, tR_e, tT_e = None, None, None, None
        else:
            dE_e, dEc_e = [ _x if _x is None else -1 for _x in freeze[:2] ]

            if tail:
                tR_e, tT_e = [ _x if _x is None else -1 for _x in freeze[2:] ]
        
        if error:
            try:
                import numdifftools as nd

                hess = nd.Hessian(stat)
                # Somehow we need x2 for the inversed hess to get the right error
                err = np.sqrt(np.diag(np.linalg.inv(hess(x))*2))

                if freeze is None:
                    dE_e, dEc_e = err[:2]
            
                    if tail:
                        tR_e, tT_e = err[2:]
                    else:
                        tR_e, tT_e = 0, 0
                else:
                    _err = list(err)
            
                    dE_e, dEc_e = [ _err.pop(0) if _x is None else -1 for _x in freeze[:2] ]
        
                    if tail:
                        tR_e, tT_e = [ _err.pop(0) if _x is None else -1 for _x in freeze[2:] ]
                    else:
                        tR_e, tT_e = 0, 0
        
                dE_e, dEc_e, tR_e, tT_e = np.nan_to_num([dE_e, dEc_e, tR_e, tT_e])

            except ImportError:
                print("Warning: Plese install numdifftools to calculate standard error.")
                
                # No error
                dE_e, dEc_e, tR_e, tT_e = 0, 0, 0, 0
                
            except np.linalg.LinAlgError:
                pass
    
    if filename is not None:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = str(tex)
        from matplotlib import pyplot as plt

        plt.figure(figsize=(8, 6))
        
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        if dE < 1:
            text = r'FWHM$=%.4f$' % dE + (' eV' if dE_e is None else ' eV (fixed)' if dE_e < 0 else r'$\pm %.4f$ eV' % dE_e)
        else:
            text = r'FWHM$=%.2f$' % dE + (' eV' if dE_e is None else ' eV (fixed)' if dE_e < 0 else r'$\pm %.2f$ eV' % dE_e)
        if dEc < 0.1:
            text += '\n' + r'$\Delta$Ec$=%.4f$' % dEc + (' eV' if dEc_e is None else ' eV (fixed)' if dEc_e < 0 else r'$\pm %.4f$ eV' % dEc_e)
        else:
            text += '\n' + r'$\Delta$Ec$=%.2f$' % dEc + (' eV' if dEc_e is None else ' eV (fixed)' if dEc_e < 0 else r'$\pm %.2f$ eV' % dEc_e)
        if tail:
            text += '\n' + r'Low-$E$ tail:'
            text += '\n' + r'  fraction: $%.2f$' % tR + ('' if tR_e is None else ' (fixed)' if tR_e < 0 else r'$\pm %.2f$' % tR_e)
            text += '\n' + r'  decay: $%.2f$' % tT + (' eV' if tT_e is None else ' eV (fixed)' if tT_e < 0 else r'$\pm %.2f$ eV' % tT_e)
        if method == 'c':
            text += '\n' + r'c-stat = $%.1f$' % s
            text += '\n' + r'd.o.f. = $%d$' % dof
        elif method == 'cs':
            text += '\n' + r'Reduced $\chi^2$ = %.1f/%d = %.2f' % (s, dof, s/dof)
        text += '\n' + r'$%d$ counts' % len(pha)

        ax1.errorbar(bincenters, ngn, yerr=ngn_sigma, xerr=np.diff(gbins)/2, capsize=0, ecolor='k', fmt='None')

        E = np.linspace(bins.min(), bins.max(), 1000)
        m = len(pha)*line_model(E, dE, dEc, tR, tT, line=line, shift=shift, tail=tail, full=True)

        # Plot theoretical model
        ax1.plot(E, m[0], 'r-')

        # Plot fine structures
        if len(m) > 2:
            ax1.plot(E, m[1:].T, 'b--')

        ax1.set_ylabel(r'Normalized Counts/eV')
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*1.1)
        ax1.text(0.02, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)
        
        ax1.ticklabel_format(axis='x', style='plain')
        xtl1 = ax1.get_xticklabels()
        plt.setp(xtl1, visible=False)
        
        # Plot residuals
        m = len(pha)*line_model(bincenters, dE, dEc, tR, tT, line=line, shift=shift, tail=tail)
        
        ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
        
        ax2.axhline(y=0, c='r')
        ax2.errorbar(bincenters, (ngn-m)/ngn_sigma, yerr=1, xerr=np.diff(gbins)/2, capsize=0, ecolor='k', fmt='None')
        ax2.set_xlabel(r'Energy$\quad$(eV)')
        ax2.set_ylabel(r'$\Delta/\sqrt{\lambda}$')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(filename)
    
    if tail:
        return (dE, dEc, tR, tT), (dE_e, dEc_e, tR_e, tT_e), (s, dof)
    else:
        return (dE, dEc), (dE_e, dEc_e), (s, dof)

