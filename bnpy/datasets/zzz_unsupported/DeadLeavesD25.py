import DeadLeaves as DL
from DeadLeaves import get_data, get_short_name, get_data_info

DL.makeTrueParams(25)

if __name__ == '__main__':
    DL.plotTrueCovMats(doShowNow=False)
    DL.plotImgPatchPrototypes()
