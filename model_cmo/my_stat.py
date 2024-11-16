import os

import scipy
from scipy.interpolate import griddata
import numpy as np
from sys import path
import pickle
from iminuit import Minuit
import matplotlib.pyplot as plt
from reverse_systs import reverse_parameterize_systs
import scipy.special

path.append("../")
path.append("../ingestion_program")


class StatisticalAnalysis:

    def __init__(self, model, bins=10, systematics=None):
        self.model = model
        self.bins = bins
        # self.bin_edges = np.linspace(0, 1, bins + 1)
        self.bin_edges = np.linspace(0.8, 1, bins + 1)
        self.syst_settings = {
            'tes': 1.0,
            'bkg_scale': 1.0,
            'jes': 1.0,
            'soft_met': 0.0,
            'ttbar_scale': 1.0,
            'diboson_scale': 1.0,
        }
        self.systematics = systematics  # Function to apply systematics

        # TODO: fix mean and std
        self.alpha_ranges = {
            "tes": {
                # TODO: refine the range
                # "range": np.linspace(0.9, 1.1, 15),
                "range": np.linspace(0.9, 1.1, 3),
                "mean": 1.0,
                "std": 0.01,
            },
            "bkg_scale": {
                "range": np.linspace(0.99, 1.01, 15),
                "mean": 1.0,
                "std": 0.001,
            },
            "jes": {
                # TODO: refine the range
                # "range": np.linspace(0.9, 1.1, 15),
                "range": np.linspace(0.9, 1.1, 3),
                "mean": 1.0,
                "std": 0.01,
            },
            "soft_met": {
                "range": np.linspace(0.0, 5.0, 15),
                "mean": 0.0,
                # TODO: soft_met should be positive
                "std": 1.0,
            },
            "ttbar_scale": {
                "range": np.linspace(0.8, 1.2, 15),
                "mean": 1.0,
                "std": 0.02,
            },
            "diboson_scale": {
                "range": np.linspace(0.0, 2.0, 15),
                "mean": 1.0,
                "std": 0.25,
            },
        }

        self.run_syst = None

        self.template_bin = 1000
        self.template_bin_edges = np.linspace(0, 1, self.template_bin + 1)
        self.template_sig = {
            'nominal': None,
            **{
                syst: {'up': None, 'down': None} for syst in ['soft_met'] # only soft_met will be considered in the template
            }
        }
        self.template_ttbar = {
            'nominal': None,
            **{
                syst: {'up': None, 'down': None} for syst in ['soft_met']
            }
        }
        self.template_diboson = {
            'nominal': None,
            **{
                syst: {'up': None, 'down': None} for syst in ['soft_met']
            }
        }
        self.template_other_bkg = {
            'nominal': None,
            **{
                syst: {'up': None, 'down': None} for syst in ['soft_met']
            }
        }
        self.template_all_bkg = {
            'nominal': None,
            **{
                syst: {'up': None, 'down': None} for syst in self.alpha_ranges.keys()
            }
        }


    def build_data_templates(self, data, weights=None):
        # build data template for each score bin. The template is the sow as a function of tes and jes

        # only consider nominal score> self.bin_edges[0]*0.6
        data_nominal = reverse_parameterize_systs(data)
        score_nominal = self.model.predict(data_nominal)

        mask = score_nominal > self.bin_edges[0]*0.6
        data_slimed = data[mask].copy()
        weights_slimed = weights[mask].copy() if weights is not None else None

        tes_range, jes_range = self.alpha_ranges['tes']['range'], self.alpha_ranges['jes']['range']
        histograms = np.zeros((self.bins, len(tes_range), len(jes_range)))

        template_x, template_y = [], []
        for i_tes, tes in enumerate(tes_range):
            for i_jes, jes in enumerate(jes_range):
                if tes == 1.0 and jes == 1.0:
                    hist = np.histogram(score_nominal, bins=self.bin_edges, density=False, weights=weights)[0]
                else:
                    data_reversed = reverse_parameterize_systs(data_slimed.copy(), tes, jes)
                    score = self.model.predict(data_reversed)
                    hist, bins = np.histogram(score, bins=self.bin_edges, density=False, weights=weights_slimed)
                histograms[:, i_tes, i_jes] = hist
                template_x.append(tes)
                template_y.append(jes)
        
        # build the template for each score bin
        templates = [None for _ in range(self.bins)]

        def build_template(hist_vs_tesid_jesid):
            def template(tes, jes):
                return griddata((template_x, template_y), hist_vs_tesid_jesid.flatten(), (tes, jes), method='linear')
            return template

        for i in range(self.bins):
            templates[i] = build_template(histograms[i])
        
        return templates


    def compute_mu(self, observed_data, weight_data):
        self.bins = 101
        # self.bins = 20
        self.bin_edges = np.linspace(0.8, 1, self.bins + 1) # or np.linspace(0.801, 1, 101)
        print(f"Number of bins: {self.bins}", flush=True)

        # TODO: statistic test. Fix this
        data_templates = self.build_data_templates(observed_data, weight_data)

        def get_yields(template, alpha):
            yields = self.rebin_hist(template['nominal'], self.template_bin_edges, self.bin_edges)
            for syst in template.keys():
                if syst=='nominal': continue
                yields += (alpha[syst] - self.alpha_ranges[syst]['mean']) * (
                        self.rebin_hist(template[syst]['up'], self.template_bin_edges, self.bin_edges) -
                        self.rebin_hist(template[syst]['down'], self.template_bin_edges, self.bin_edges)
                ) / 2

            return yields

        overall_norm_factor = 1. # TODO: remove this
        def NLL(mu, tes, bkg_scale, jes, soft_met, ttbar_scale, diboson_scale):
            # TODO: statistic test. Fix this
            obs_hist = np.zeros(self.bins)
            for i in range(self.bins):
                obs_hist[i] = data_templates[i](tes, jes)

            alpha = {
                'tes': tes,
                'bkg_scale': bkg_scale,
                'jes': jes,
                'soft_met': soft_met,
                'ttbar_scale': ttbar_scale,
                'diboson_scale': diboson_scale,
            }

            exp_hist = mu * get_yields(self.template_sig, alpha) +  \
                    bkg_scale * ttbar_scale * get_yields(self.template_ttbar, alpha) + \
                    bkg_scale * diboson_scale * get_yields(self.template_diboson, alpha) + \
                    bkg_scale * get_yields(self.template_other_bkg, alpha)
            exp_hist *= overall_norm_factor

            epsilon = 1e-10
            exp_hist = np.clip(exp_hist, epsilon, None)

            nll_poisson = exp_hist - obs_hist * np.log(exp_hist) + scipy.special.gammaln(obs_hist + 1)

            nll_gauss = 0.5 * np.sum([( (alpha[syst] - self.alpha_ranges[syst]['mean']) / self.alpha_ranges[syst]['std']) ** 2 for syst in alpha.keys()])


            return (nll_poisson.sum() + nll_gauss)

        result = Minuit(NLL,
                        mu=1.0,
                        tes=1.0,
                        bkg_scale=1.0,
                        jes=1.0,
                        soft_met=0.0,
                        ttbar_scale=1.0,
                        diboson_scale=1.0,
                        )

        for key, value in self.alpha_ranges.items():
            result.limits[key] = (value['range'][0], value['range'][-1])
        result.limits['mu'] = (0, 3)


        result.errordef = Minuit.LIKELIHOOD
        result.migrad()

        if not result.fmin.is_valid:
            print("Warning: migrad did not converge. Hessian errors might be unreliable.")

        error_bar_scale = 2.5
        mu_hat = result.values['mu']
        delta_mu_hat = result.errors['mu'] * error_bar_scale
        mu_p16 = mu_hat - delta_mu_hat
        mu_p16 = 0 if mu_p16 < 0 else mu_p16 
        mu_p84 = mu_hat + delta_mu_hat
        mu_p84 = 3 if mu_p84 > 3 else mu_p84 


        # print(f"mu_hat: {mu_hat:.3f}, delta_mu_hat: {delta_mu_hat * 2:.3f}, p16: {mu_p16:.3f}, p84: {mu_p84:.3f}")
        print(f"tes: {result.values['tes']:.3f}, jes: {result.values['jes']:.3f}, soft_met: {result.values['soft_met']:.3f}, ttbar_scale: {result.values['ttbar_scale']:.3f}, diboson_scale: {result.values['diboson_scale']:.3f}, bkg_scale: {result.values['bkg_scale']:.3f}", flush=True)

        return {
            "mu_hat": mu_hat,
            "delta_mu_hat": mu_p84 - mu_p16 ,
            "p16": mu_p16,
            "p84": mu_p84,
        }

    def calculate_template(self, holdout_set, file_path):
        holdout_set["data"].reset_index(drop=True, inplace=True)

        def get_distribution(syst_name, syst_value):
            syst_settings = self.syst_settings.copy()

            if syst_name in self.syst_settings:
                syst_settings[syst_name] = syst_value

            holdout_syst = self.systematics(
                holdout_set.copy(),
                **syst_settings
            )

            label_holdout = holdout_syst['labels'].to_numpy()
            weights_holdout = holdout_syst['weights'].to_numpy()
            detailed_labels_holdout = holdout_syst['detailed_labels'].to_numpy()

            # reverse the systematics
            holdout_data_reversed = reverse_parameterize_systs(holdout_syst['data'].copy(), **syst_settings)

            holdout_val = self.model.predict(holdout_data_reversed)

            histograms = []
            for selection_criteria in [
                "label_holdout == 1", # signal
                'detailed_labels_holdout == "ttbar"', # ttbar
                'detailed_labels_holdout == "diboson"', # diboson
                '(label_holdout == 0) & (detailed_labels_holdout != "ttbar") & (detailed_labels_holdout != "diboson")', # other background
            ]:
                hist, bins = np.histogram(holdout_val[eval(selection_criteria)],
                                        bins=self.template_bin_edges, density=False,
                                        weights=weights_holdout[eval(selection_criteria)])
                histograms.append(hist)
            
            holdout_signal_hist, holdout_ttbar_hist, holdout_diboson_hist, holdout_other_bkg_hist = histograms
            return holdout_signal_hist, holdout_ttbar_hist, holdout_diboson_hist, holdout_other_bkg_hist


        # Calculate nominal template
        self.template_sig['nominal'], self.template_ttbar['nominal'], self.template_diboson['nominal'], self.template_other_bkg['nominal'] = get_distribution(None, None)
        self.template_all_bkg['nominal'] = self.template_ttbar['nominal'] + self.template_diboson['nominal'] + self.template_other_bkg['nominal']

        # Calculate systematic templates
        os.makedirs('plots-mystat/systematics', exist_ok=True)

        # TODO: In principle, only soft_met need to be considered in the template
        for syst in ['soft_met']:
            if syst == 'nominal': continue
            self.template_sig[syst]['up'], self.template_ttbar[syst]['up'], self.template_diboson[syst]['up'], self.template_other_bkg[syst]['up'] = get_distribution(
                syst, self.alpha_ranges[syst]['mean'] + self.alpha_ranges[syst]['std']
            )

            self.template_sig[syst]['down'], self.template_ttbar[syst]['down'], self.template_diboson[syst]['down'], self.template_other_bkg[syst]['down'] = get_distribution(
                syst, self.alpha_ranges[syst]['mean'] - self.alpha_ranges[syst]['std']
            )

            self.template_all_bkg[syst]['up'] = self.template_ttbar[syst]['up'] + self.template_diboson[syst]['up'] + self.template_other_bkg[syst]['up']
            self.template_all_bkg[syst]['down'] = self.template_ttbar[syst]['down'] + self.template_diboson[syst]['down'] + self.template_other_bkg[syst]['down']

            self.plot_systematics_effect(
                self.template_sig, self.template_all_bkg, syst, save_name=f'plots-mystat/systematics/{syst}.png'
            )
            print(syst)
        
        print(self.template_sig.keys())

        with open(os.path.join(file_path, f'template.pkl'), "wb") as f:
            pickle.dump({
                'sig': self.template_sig,
                'ttbar': self.template_ttbar,
                'diboson': self.template_diboson,
                'other_bkg': self.template_other_bkg
            }, f)


    def load(self, file_path):
        """
        Load the saved_info dictionary from a file.

        Args:
            file_path (str): File path to load the object.

        Returns:
            None
        """
        f = os.path.join(file_path, f"template.pkl")

        if not os.path.exists(f):
            return False

        with open(os.path.join(file_path, f"template.pkl"), "rb") as f:
            template = pickle.load(f)
            self.template_sig = template['sig']
            self.template_ttbar = template['ttbar']
            self.template_diboson = template['diboson']
            self.template_other_bkg = template['other_bkg']

            return True

    def rebin_hist(self, hist, old_bin_edges, new_bin_edges):
        """
        Rebin a histogram to match new bin edges.

        Parameters:
        hist: array-like
            The counts from np.histogram (hist[0]).
        old_bin_edges: array-like
            The original bin edges from np.histogram (hist[1]).
        new_bin_edges: array-like
            The new bin edges to rebin the histogram to.

        Returns:
        rebinned_hist: array-like
            The rebinned counts.
        """
        # Create an array to store the rebinned histogram counts
        rebinned_hist = np.zeros(len(new_bin_edges) - 1)

        # Loop over new bins
        for i in range(len(new_bin_edges) - 1):
            # Find the old bins that overlap with the new bin
            in_bin = (old_bin_edges[:-1] >= new_bin_edges[i]) & (old_bin_edges[:-1] < new_bin_edges[i + 1])

            # Sum the counts from the old bins that fall within the new bin range
            rebinned_hist[i] = np.sum(hist[in_bin])

        return rebinned_hist

    def plot_systematics_effect(self, template_sig, template_bkg, syst, save_name=None):

        fig, ax = plt.subplots(figsize=(10, 6))

        # Signal histograms: nominal, +1σ, -1σ
        nominal_sig = template_sig['nominal']
        up_sig = template_sig[syst]['up']
        down_sig = template_sig[syst]['down']

        # Background histograms: nominal, +1σ, -1σ
        nominal_bkg = template_bkg['nominal']
        up_bkg = template_bkg[syst]['up']
        down_bkg = template_bkg[syst]['down']

        # rebin histograms for better visualization
        bins = np.linspace(0, 1, 100)
        nominal_sig = self.rebin_hist(nominal_sig, self.template_bin_edges, bins)
        up_sig = self.rebin_hist(up_sig, self.template_bin_edges, bins)
        down_sig = self.rebin_hist(down_sig, self.template_bin_edges, bins)
        nominal_bkg = self.rebin_hist(nominal_bkg, self.template_bin_edges, bins)
        up_bkg = self.rebin_hist(up_bkg, self.template_bin_edges, bins)
        down_bkg = self.rebin_hist(down_bkg, self.template_bin_edges, bins)


        print(f"[{syst}] Sig: {nominal_sig.sum()}, {up_sig.sum()}, {down_sig.sum()}")
        print(f"[{syst}] Bkg: {nominal_bkg.sum()}, {up_bkg.sum()}, {down_bkg.sum()}")

        # Calculate normalization factor based on nominal histograms
        NF_sig = np.sum(nominal_sig)
        NF_bkg = np.sum(nominal_bkg)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10

        # Normalize signal histograms using NF_sig
        nominal_sig = nominal_sig / (NF_sig + epsilon)
        up_sig = up_sig / (NF_sig + epsilon)
        down_sig = down_sig / (NF_sig + epsilon)

        # Normalize background histograms using NF_bkg
        nominal_bkg = nominal_bkg / (NF_bkg + epsilon)
        up_bkg = up_bkg / (NF_bkg + epsilon)
        down_bkg = down_bkg / (NF_bkg + epsilon)

        # Plot nominal, +1σ, -1σ for signal
        ax.plot(nominal_sig, label=f'Signal (nominal)', color='blue', lw=2)
        ax.plot(up_sig, label=f'Signal (+1σ)', color='green', linestyle='--')
        ax.plot(down_sig, label=f'Signal (-1σ)', color='red', linestyle='--')

        # Plot nominal, +1σ, -1σ for background
        ax.plot(nominal_bkg, label=f'Background (nominal)', color='blue', lw=2, alpha=0.5)
        ax.plot(up_bkg, label=f'Background (+1σ)', color='green', linestyle=':', alpha=0.7)
        ax.plot(down_bkg, label=f'Background (-1σ)', color='red', linestyle=':', alpha=0.7)

        # Calculate overall effect for signal and background separately
        overall_effect_sig_up = np.sum(up_sig - nominal_sig) / np.sum(nominal_sig + 1e-8)
        overall_effect_sig_down = np.sum(down_sig - nominal_sig) / np.sum(nominal_sig + 1e-8)

        overall_effect_bkg_up = np.sum(up_bkg - nominal_bkg) / np.sum(nominal_bkg + 1e-8)
        overall_effect_bkg_down = np.sum(down_bkg - nominal_bkg) / np.sum(nominal_bkg + 1e-8)

        # Add overall effect for signal to the plot as text
        ax.text(0.05, 0.95, f'Signal Effect (+1σ): {overall_effect_sig_up:.5%}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='green')
        ax.text(0.05, 0.90, f'Signal Effect (-1σ): {overall_effect_sig_down:.5%}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='red')

        # Add overall effect for background to the plot as text
        ax.text(0.05, 0.85, f'Background Effect (+1σ): {overall_effect_bkg_up:.5%}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='green', alpha=0.7)
        ax.text(0.05, 0.80, f'Background Effect (-1σ): {overall_effect_bkg_down:.5%}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='red', alpha=0.7)
        # Labels and title
        ax.set_xlabel('Bins')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()

        if save_name:
            fig.savefig(save_name, dpi=300)

    def plot_stacked_histogram(self, bins, signal_fit, background_fit, mu, N_obs, save_name=None):
        """
        Plot a stacked histogram with combined signal and background fits and observed data points.

        Parameters:
            bins (numpy.ndarray): Bin edges.
            signal_fit (numpy.ndarray): Combined signal fit values.
            background_fit (numpy.ndarray): Combined background fit values.
            mu (float): Multiplicative factor for the signal.
            N_obs (numpy.ndarray): Observed data points.
            save_name (str, optional): Name of the file to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), gridspec_kw={'height_ratios': [2, 1]})

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_widths = np.diff(bins)

        # Main plot: stacked histograms for signal and background
        ax1.bar(bin_centers, signal_fit, width=bin_widths, color='g', align='center', label='Signal')
        ax1.bar(bin_centers, background_fit, width=bin_widths, alpha=0.5, label='Background', color='b', align='center')
        ax1.bar(bin_centers, signal_fit * mu, width=bin_widths, alpha=0.5, label=f'Signal * {mu:.1f}', color='r',
                align='center', bottom=background_fit)

        # Plot observed data points
        ax1.errorbar(bin_centers, N_obs, yerr=np.sqrt(N_obs), fmt='o', color='k', label='Observed Data')

        ax1.set_xlabel('Score')
        ax1.set_ylabel('Counts')
        ax1.set_yscale('log')  # Set y-axis to logarithmic scale
        ax1.set_title('Stacked Histogram: Signal and Background Fits with Observed Data')
        ax1.legend()

        # Ratio plot: N_obs / (background_fit + signal_fit)
        expected = background_fit + signal_fit
        ratio = N_obs / (expected + 1e-8)
        rel_error = (1 / (N_obs + 1e-8) + 1 / (expected + 1e-8))**0.5
        error = (rel_error) * ratio
        ax2.errorbar(bin_centers, ratio, yerr=error, fmt='o', color='k', label='N_obs / expected')
        ax2.axhline(1, color='r', linestyle='--', label='Expected Ratio = 1')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Ratio')
        ax2.set_ylim(0.9, 1.1)
        ax2.legend()

        # # Subplot: distribution of (N_obs - background_fit) and signal_fit
        # diff = N_obs - background_fit
        # ax2.errorbar(bin_centers, diff, yerr=np.sqrt(np.abs(diff)), fmt='o', color='k', label='N_obs - Background')

        # # ax2.bar(bin_centers, diff, width=bin_widths, color='purple', align='center', label='N_obs - Background')
        # ax2.bar(bin_centers, signal_fit, width=bin_widths, color='g', align='center', alpha=0.5, label='Signal')

        # ax2.set_xlabel('Score')
        # ax2.set_ylabel('Counts')
        # ax2.legend()


        plt.tight_layout()
        # plt.show()

        if save_name:
            fig.savefig(save_name, dpi=300)