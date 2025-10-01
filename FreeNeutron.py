from isotope_class import isotope

# python script defining radioactive isotope data and writing to either an hdf5 or an ascii file
# data comes from nuclear data sheets, compiled at www.nndc.bnl.gov and www-nds.iaea.org
# ignoring neutrino and X-ray contribution

## Format for each data entry
# iso.name             = name of isotope in the format Z.A
# iso.source           =  # reference for data source
# iso.daughter_Z       = daughter's Z atomic number
# iso.daughter_A       = daughter's A atomic mass
# iso.t_halflife_in_s  = half-life in seconds
# iso.e_decay_in_keV   = average energy per decay in keV, including lepton KE a and positron annihilation photons, ignoring neutrino energy and X-rays
# iso.f_lepton_KE     = fraction of decay energy from lepton KE (positrons and Auger and internal conversion electrons)
# iso.n_gamma         = number of elements in q_gamma_in_keV and i_gamma list of gamma-rays emitted
# iso.q_gamma_in_keV  = list of energies of emitted gamma rays in keV, including positron annihilation photons
# iso.i_gamma         = percent incidence per each decay (e.g., 100 is 100%)

# output filename
fname = 'free_neutron_radioactive_data.h5'

#Free Neutron
iso = isotope()
iso.name            = '0.1'
iso.source          = 'Particle Data Group (2007). Summary Data Table on Baryons. Lawrence Berkeley Laboratory. and Kulkarni 2005 (arXiv:astro-ph/0510256)'
iso.daughter_Z      = 1
iso.daughter_A      = 1
iso.t_halflife_in_s = 613.9
iso.e_decay_in_keV  = 302.772032134
iso.f_lepton_KE     = 1.0
iso.n_gamma         = 0
iso.q_gamma_in_keV  = []
iso.i_gamma         = []
iso.add_to_file(fname)

#Stable Hydrogen
iso = isotope()
iso.name            = '1.1'
iso.source          = ''
iso.daughter_Z      = 0
iso.daughter_A      = 0
iso.t_halflife_in_s = 9.0e99
iso.e_decay_in_keV  = 0.0
iso.f_lepton_KE     = 0.0
iso.n_gamma         = 0
iso.q_gamma_in_keV  = []
iso.i_gamma         = []
iso.add_to_file(fname)
