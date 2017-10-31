from builtins import *
import numpy as np


def flatstr2np(xvecstr):
    return np.asarray([float(x) for x in xvecstr.split()])


def np2flatstr(arr, fmt="% .6g"):
    return np.array2string(
        arr,
        prefix='',
        separator=' ',
        formatter={'float_kind':lambda x: fmt % x})
    #    return ' '.join([fmt % (x) for x in np.asarray(X).flatten()])


def np2strList(X, fmt="%.4f", zeroThr=1e-25, zeroSymb=''):
    slist = list()
    for x in np.asarray(X).flatten():
        if np.abs(x) < zeroThr:
            s = zeroSymb
        else:
            s = fmt % (x)
        if np.unique(s[2:]).size == 1 and s.startswith('0.'):
            s = s[1:]
            s = s[:-1] + '1'
            s = '<' + s
        slist.append(s)
    return slist


def split_str_into_fixed_width_lines(mstr, linewidth=80, tostr=False):
    ''' Split provided string across lines nicely.

    Examples
    --------
    >>> s = ' abc def ghi jkl mno pqr'
    >>> split_across_lines(s, linewidth=5)
    >>> split_across_lines(s, linewidth=7)
    >>> split_across_lines(s, linewidth=10)
    >>> s = '   abc   def   ghi   jkl   mno   pqr'
    >>> split_across_lines(s, linewidth=5)
    >>> split_across_lines(s, linewidth=7)
    >>> split_across_lines(s, linewidth=10)
    >>> s = '  abc1  def2  ghi3  jkl4'
    >>> split_across_lines(s, linewidth=3)
    >>> split_across_lines(s, linewidth=6)
    >>> split_across_lines(s, linewidth=9)
    >>> split_across_lines(s, linewidth=80)
    '''
    mlist = list()
    breakPos = 0
    while breakPos < len(mstr):
        if (len(mstr) - breakPos) <= linewidth:
            # Take it all and quit
            mlist.append(mstr[breakPos:])
            break
        else:
            nextPos = breakPos+linewidth
            while nextPos > breakPos + 1:
                if mstr[nextPos-1] != ' ' and mstr[nextPos] == ' ':
                    break
                nextPos -= 1
            nextstr = mstr[breakPos:nextPos]
            if len(nextstr.strip()) > 0:
                mlist.append(nextstr)
            breakPos = nextPos
    mlist[0] = ' ' + mlist[0] # hack
    if tostr:
        return '\n'.join([m for m in mlist])
    return mlist
