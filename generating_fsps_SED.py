import fsps
import numpy as np
import pandas as pd
import sys, os
zred = float(sys.argv[1])
sp = fsps.StellarPopulation(zcontinuous=1, add_neb_emission=1)
sp.params['logzsol'] = -1.0
sp.params['gas_logz'] = -1.0
sp.params['gas_logu'] = -2.5

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)

tuniv = cosmo.age(zred).to('Gyr').value


wave, spec = sp.get_spectrum(tage=tuniv)



df = pd.DataFrame({'wav':wave, 'spec':spec})
print(df)
df.to_csv(f'SED_zred{zred}.csv', index=False)
