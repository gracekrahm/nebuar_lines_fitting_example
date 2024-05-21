import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sedpy.observate import load_filters
from astropy import units as u
import prospect.io.read_results as reader

mpl.use('Tkagg')

import sys
def get_best(res, **kwargs):
    """Get the maximum a posteriori parameters.                                                                                                               
    From prospect.utils.plotting                                                                                                                              
    """
    imax = np.argmax(res['lnprobability'])
    theta_best = res['chain'][imax, :].copy()

    return theta_best

zred = float(sys.argv[1])

infile = f'fitted_SED_zred8.h5'


def plot_sed(infile, infile_label, sedfile_label):
	print('importing model and sps')
	sys.path.append('./')
	from run_prosp import build_model, build_sps
	results, obs, model = reader.results_from(infile)

	mod_sim = model
	res = results
	try:
		sps_sim = reader.get_sps(res)
		sim_wav= sps_sim.wavelengths
	except:
		sps_sim = build_sps()
		sim_wav = sps_sim.wavelengths*(1+zred)

	thetas_sim = get_best(res)

	try:

		spec_sim, _, _ = model.mean_model(thetas_sim, res['obs'], sps_sim)
	except:
		model = build_model()
		spec_sim, _, _ = model.mean_model(thetas_sim, res['obs'], sps_sim)

	obs = res['obs']
	maggies = obs['pd_sed']
	redshifted_wav = obs['pd_wav']

	plt.loglog(sim_wav, spec_sim, label=infile_label)
	plt.loglog(redshifted_wav, maggies, label=sedfile_label, linestyle='dotted')

fig, ax = plt.subplots(figsize=(10,5))
plot_sed(infile, 'Prospector Fitted SED', 'Powderday Mock SED')




ax.set_xlabel('micron')
ax.set_ylabel('Flux (mJy)')
plt.xlim([10**3, 10**6])
plt.ylim([10**(-11), 10**(-5)])
ax.grid()
plt.legend()

plt.title(f'Redshift {zred} Galaxy SED')
#plt.show()
plt.savefig(f'fitted_sed_comparison.png')
