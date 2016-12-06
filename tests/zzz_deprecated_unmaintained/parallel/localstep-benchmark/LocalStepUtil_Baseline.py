
def calcLocalParamsAndSummarize(
        Data=None, hmodel=None, 
        LPkwargs=dict(),
        **kwargs):
    ''' calcLocalParamsAndSummarize, a non-parallel version
    '''
    LP = hmodel.calc_local_params(Data, **LPkwargs)
    SS = hmodel.get_global_suff_stats(Data, LP, **LPkwargs)
    return SS
