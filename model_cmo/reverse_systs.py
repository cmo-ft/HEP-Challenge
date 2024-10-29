import numpy as np
from derived_quantities import DER_data
"""
Mannually reverse the systematics
"""
def reverse_tes_syst(data_df, tes):
    pri_had_pt, pri_had_phi = data_df['PRI_had_pt'].to_numpy(), data_df['PRI_had_phi'].to_numpy()
    pri_met, pri_met_phi = data_df['PRI_met'].to_numpy(), data_df['PRI_met_phi'].to_numpy()
    met_px, met_py = pri_met * np.cos(pri_met_phi), pri_met * np.sin(pri_met_phi)

    met_px -= (1-tes) / tes * pri_had_pt * np.cos(pri_had_phi)
    met_py -= (1-tes) / tes * pri_had_pt * np.sin(pri_had_phi)
    data_df['PRI_met'] = np.sqrt(met_px**2 + met_py**2)
    data_df['PRI_met_phi'] = np.arctan2(met_py, met_px)
    data_df['PRI_had_pt'] = pri_had_pt / tes
    return data_df


def reverse_jes_syst(data_df, jes):
    def reverse_one_jet(pt_jet, phi_jet, pt_met, phi_met, mask_jet_valid):
        reversed_pt_jet = pt_jet

        met_px, met_py = pt_met * np.cos(phi_met), pt_met * np.sin(phi_met)
        met_px -= (1-jes) / jes * pt_jet * np.cos(phi_jet)
        met_py -= (1-jes) / jes * pt_jet * np.sin(phi_jet)

        reversed_pt_jet[mask_jet_valid] = pt_jet[mask_jet_valid] / jes
        reversed_pt_met, reversed_phi_met = pt_met, phi_met
        reversed_pt_met[mask_jet_valid] = np.sqrt(met_px**2 + met_py**2)[mask_jet_valid]
        reversed_phi_met[mask_jet_valid] = np.arctan2(met_py, met_px)[mask_jet_valid]
        return reversed_pt_jet, reversed_pt_met, reversed_phi_met
    
    # leading jet
    mask_jet_valid = data_df['PRI_n_jets'] > 0
    data_df['PRI_jet_leading_pt'], data_df['PRI_met'], data_df['PRI_met_phi'] = reverse_one_jet(
        data_df['PRI_jet_leading_pt'].to_numpy(), data_df['PRI_jet_leading_phi'].to_numpy(), data_df['PRI_met'].to_numpy(), data_df['PRI_met_phi'].to_numpy(), mask_jet_valid
    )

    # subleading jet
    mask_jet_valid = data_df['PRI_n_jets'] > 1
    data_df['PRI_jet_subleading_pt'], data_df['PRI_met'], data_df['PRI_met_phi'] = reverse_one_jet(
        data_df['PRI_jet_subleading_pt'].to_numpy(), data_df['PRI_jet_subleading_phi'].to_numpy(), data_df['PRI_met'].to_numpy(), data_df['PRI_met_phi'].to_numpy(), mask_jet_valid
    )
    return data_df


def reverse_p4(data_df, tes=1.0, jes=1.0, soft_met=0.0):
    data_df = data_df.copy()
    if tes != 1.0:
        data_df = reverse_tes_syst(data_df, tes)
    if jes != 1.0:
        data_df = reverse_jes_syst(data_df, jes)
    return data_df


def reverse_parameterize_systs(
        data=None,
        tes=1.0,
        jes=1.0,
        soft_met=0.0,
        bkg_scale=1.0,
        ttbar_scale=1.0,
        diboson_scale=1.0,
):
    """
    Manage data set that has systematics applied.
    1. Reverse the TES and JES systematics
    2. Parametrize data with soft_met syst

    Args:
        * data (pd.DataFrame): A dictionary containing the data.
        * systs
        * scales will not be used in this function
        
    Returns:
        data: A dictionary containing the data.
    """
    data_new = data.copy()
    data_reversed = reverse_p4(data_new.copy(), tes, jes, soft_met)
    data_reversed_with_der = DER_data(data_reversed.copy())

    # parametrize data with soft_met syst
    data_reversed_with_der['soft_met'] = soft_met
    return data_reversed_with_der
