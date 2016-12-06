
def PlotSingleRunCounts(taskpath, compsToShow=None):
  ''' Plot active component counts for single run
  '''
  lappath = os.path.join(taskpath, 'laps.txt')
  laps = np.loadtxt(lappath)

  Counts = LoadSingleRunCounts(taskpath)

  if compsToShow is not None:
    Counts = Counts[:, compsToShow]
    pylab.plot(laps, Counts, '.-')

  else:
    global order
    if order is None:
      order = np.arange(Counts.shape[1])

    import bnpy.viz.GaussViz
    Colors = bnpy.viz.GaussViz.Colors

    for ii, _ in enumerate(order):
      color = Colors[ii % len(Colors)]
      pylab.plot(laps, Counts[:, ii], '.-', color=color)

  pylab.ylabel('usage count', fontsize=14)
  pylab.xlabel('laps thru training data', fontsize=14)
