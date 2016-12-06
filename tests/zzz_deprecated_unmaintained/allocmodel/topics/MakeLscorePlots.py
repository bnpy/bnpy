import os
import numpy as np
import glob
from bnpy.viz import PlotUtil
pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(pylab)

def makeLscorePlots():
	# Load all by name
	fNames = glob.glob("Lscore*.txt")

	JMap = dict()
	JNone = dict()
	JByBeta = dict()
	for fName in fNames:
		shortName = fName.replace("Lscore_", '')
		shortName = shortName.replace(".txt", '')
		JMap[shortName] = np.loadtxt(fName)
		if fName.count('_sort=None'):
			JNone[shortName] = JMap[shortName]
		elif fName.count('_sort=ByBeta'):
			JByBeta[shortName] = JMap[shortName]

	nrows=1
	ncols=2
	pylab.subplots(
		nrows=nrows,
		ncols=ncols, 
		figsize=(20,5))
	ax = pylab.subplot(nrows, ncols, 1)
	for ii, key in enumerate(sorted(JNone.keys())):
		pylab.plot(JNone[key], '.-',
			markersize=10 - 2 * ii,
			linewidth=2,
			label=key[:key.index("_")])
	pylab.ylabel('ELBO score')
	pylab.xlabel('iterations')
	pylab.title('Fixed order')
	pylab.legend(loc='lower right')
	ax.yaxis.grid(color='gray', linestyle='dashed')

	ax = pylab.subplot(nrows, ncols, 2, sharex=ax, sharey=ax)
	ax.yaxis.grid(color='gray', linestyle='dashed')
	for ii, key in enumerate(sorted(JByBeta.keys())):
		pylab.plot(JByBeta[key], '.-',
			markersize=10 - 2 * ii,
			linewidth=2,
			label=key[:key.index("_")])
		if ii == 0:
			ymin = np.percentile(JByBeta[key], 10)
			ymax = np.percentile(JByBeta[key], 100)
	pylab.xlabel('iterations')
	pylab.title('Sort By Beta')

	pylab.ylim([ymin, ymax + 0.05*(ymax-ymin)])
	pylab.show(block=1)

if __name__ == "__main__":
	makeLscorePlots()
