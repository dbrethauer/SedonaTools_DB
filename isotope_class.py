import h5py

class isotope:

    name            = ''      # name of isotope in the format Z.A
    source          = ''      # data source
    daughter_Z      = 0       # daughter's charge
    daughter_A      = 0       # daughter's mass
    t_halflife_in_s = 9.0e99  # half-life in seconds
    e_decay_in_keV  = 0.0     # average energy per decay in keV, including lepton KE and positron annihilation photons, ignoring neutrino energy and X-rays
    f_lepton_KE     = 0.0     # fraction of decay energy from lepton KE (positrons and Auger and internal conversion electrons)
    n_gamma         = 0       # number of elements in q_gamma_in_keV and i_gamma
    q_gamma_in_keV  = []      # list of energies of emitted gamma rays in keV, including positron annihilation photons
    i_gamma         = []      # percent incidence per each decay (e.g., 100 is 100%)

    def add_to_file(self,fname):

      f = h5py.File(fname,'a')
      base = self.name + "/"
      try:
          del f[base]
      except:
          pass
      grp = f.create_group(base)
      grp.attrs['source'] = self.source
      f.create_dataset(base + "t_halflife_in_s",data = self.t_halflife_in_s)
      f.create_dataset(base + "daughter_Z",data = self.daughter_Z)
      f.create_dataset(base + "daughter_A",data = self.daughter_A)
      dset = f.create_dataset(base + "e_decay_in_keV",data=self.e_decay_in_keV)
      dset.attrs['description'] = 'average energy per decay in keV, including lepton KE and positron annihilation photons, ignoring neutrino energy and X-rays'
      dset = f.create_dataset(base + "f_lepton_KE",data=self.f_lepton_KE)
      dset.attrs['description'] = 'fraction of decay energy from lepton KE (positrons and Auger and internal conversion electrons)'
      dset = f.create_dataset(base + "n_gamma",data=self.n_gamma)
      dset.attrs['description'] = 'number of elements in q_gamma_in_keV and i_gamma'
      dset = f.create_dataset(base + "q_gamma_in_keV",data=self.q_gamma_in_keV)
      dset.attrs['description'] = 'list of energies of emitted gamma rays in keV, including positron annihilation photons'
      dset = f.create_dataset(base + "i_gamma",data=self.i_gamma)
      dset.attrs['description'] = 'percent incidence per each decay (e.g., 100 is 100%)'
      f.close()
