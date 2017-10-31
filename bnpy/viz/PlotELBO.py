from builtins import *
from .PlotTrace import plotJobsThatMatchKeywords, plotJobs, parse_args

plotJobsThatMatch = plotJobsThatMatchKeywords

if __name__ == "__main__":
    argDict = parse_args(xvar='laps', yvar='evidence')
    plotJobsThatMatchKeywords(**argDict)
