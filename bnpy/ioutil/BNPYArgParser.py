import argparse
import ConfigParser
import os
import sys
import numpy as np

from bnpy.allocmodel import AllocModelNameSet
from bnpy.obsmodel import ObsModelNameSet

FullDataAlgSet = set(['EM', 'VB', 'GS', 'pVB'])
OnlineDataAlgSet = set(['soVB', 'moVB', 'memoVB', 'pmoVB'])
algChoices = FullDataAlgSet | OnlineDataAlgSet

KwhelpHelpStr = "Include --kwhelp to print our keyword argument help and exit"

dataHelpStr = "Name of dataset or dataset object." \
    + " Name can point to python script in $BNPYDATADIR"

choiceStr = ' {' + ','.join([x for x in (AllocModelNameSet)]) + '}'
aModelHelpStr = 'Name of allocation model.' + choiceStr

choiceStr = ' {' + ','.join([x for x in (ObsModelNameSet)]) + '}'
oModelHelpStr = 'Name of observation model.' + choiceStr

choiceStr = ' {' + ','.join([x for x in (algChoices)]) + '}'
algHelpStr = 'Name of learning algorithm.' + choiceStr

MovesHelpStr = """String names of moves to perform.
    Options: {birth,merge,delete}.
    To perform several move types separate them with commas,
     like 'birth,merge' or 'delete,birth'. Do not include spaces."""



def parseRequiredArgs():
    ''' Process standard input for bnpy's required arguments, as a dict.

    Required Args: dataName, allocModelName, obsModelName, algName

    All other args in stdin are left alone (for later processing).

    Returns
    --------
    ArgDict : dict, with fields
    * dataName
    * allocModelName
    * obsModelName
    * algName
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName', type=str, help=dataHelpStr)
    parser.add_argument('allocModelName',
                        type=str, help=aModelHelpStr)
    parser.add_argument('obsModelName',
                        type=str, help=oModelHelpStr)
    parser.add_argument('algName',
                        type=str, help=algHelpStr)
    args, unk = parser.parse_known_args()
    if args.allocModelName not in AllocModelNameSet:
        raise ValueError('Unrecognized allocModelName %s' %
                         (args.allocModelName))
    if args.obsModelName not in ObsModelNameSet:
        raise ValueError('Unrecognized obsModelName %s' % (args.obsModelName))
    if args.algName not in algChoices:
        raise ValueError('Unrecognized learning algName %s' % (args.algName))
    return args.__dict__


def parseKeywordArgs(ReqArgs, **kwargs):
    ''' Process standard input or provided input for keyword args.

    If kwargs are provided, this is what we process.
    Otherwise, we read stdin into a kwargs dict, and process that.

    Default keys and values for each possible option
    are read in from .conf config files in $BNPYROOT/bnpy/config/

    Returns
    --------
    KwDict : dict
        Has key/val pair for every option defined in config files.
    UnkDict : dict
        Contains all remaining key/value pairs in provided kwargs
        that were not recognized options defined in config files.
    '''
    movesParser = argparse.ArgumentParser()
    movesParser.add_argument(
        '--moves', type=str, default=None, help=MovesHelpStr)
    MovesArgDict, unkDict = applyParserToStdInOrKwargs(movesParser, **kwargs)

    Moves = set()
    if MovesArgDict['moves'] is not None:
        for move in MovesArgDict['moves'].split(','):
            Moves.add(move)

    # Create parser, fill with default options from files
    parser = makeParserWithDefaultsFromConfigFiles(ReqArgs, Moves)
    parser.add_argument('--kwhelp', action='store_true', help=KwhelpHelpStr)

    # Apply the parser to input keywords
    kwargs, unkDict = applyParserToStdInOrKwargs(parser, **kwargs)
    if kwargs['kwhelp']:
        parser.print_help()
        sys.exit(-1)

    if 'moves' in unkDict:
        del unkDict['moves']

    # Transform kwargs from "flat" dict, with no sense of sections
    # into a multi-level dict, with section names like
    # 'EM', 'Gauss', 'MixModel', 'Init', etc.
    kwargs = organizeParsedArgDictIntoSections(ReqArgs, Moves, kwargs)
    kwargs['MoveNames'] = [movestr for movestr in Moves]
    return kwargs, unkDict


def applyParserToStdInOrKwargs(parser, **kwargs):
    ''' Extract all fields defined in parser from stdin (or provided kwargs)

    If no kwargs provided, they are read from stdin.

    Returns
    --------
    ArgDict : dict
        Parsed key/value pairs for all fields defined in parser
    UnkDict : dict
        Key/value pairs for any keys unknown by the parser.
    '''
    if len(kwargs.keys()) > 0:
        kwlist, ComplexTypeDict = kwargs_to_arglist(**kwargs)
        args, unkList = parser.parse_known_args(kwlist)
        ArgDict = args.__dict__
        ArgDict.update(ComplexTypeDict)
    else:
        # Parse all args/kwargs from stdin
        args, unkList = parser.parse_known_args()
        ArgDict = args.__dict__
    return ArgDict, arglist_to_kwargs(unkList)


def arglist_to_kwargs(alist, doConvertFromStr=True):
    ''' Transform list into key/val pair dictionary

    Neighboring entries in list are interpreted as key/value pairs.

    Returns
    -------
    kwargs : dict
        Each value is cast to appropriate primitive type
        (float/int/str) if possible. Complicated types are left alone.

    Examples
    ---------
    >>> arglist_to_kwargs(['--a', '1', '--b', 'stegosaurus'])
    {'a': 1, 'b': 'stegosaurus'}
    >>> arglist_to_kwargs(['requiredarg', 1])
    {}
    '''
    kwargs = dict()
    a = 0
    while a + 1 < len(alist):
        curarg = alist[a]
        if curarg.startswith('--'):
            argname = curarg[2:]
            argval = alist[a + 1]
            if isinstance(argval, str) and doConvertFromStr:
                curType = _getTypeFromString(argval)
                kwargs[argname] = curType(argval)
            else:
                kwargs[argname] = argval
            a += 1
        a += 1
    return kwargs


def kwargs_to_arglist(**kwargs):
    ''' Transform dict key/value pairs into an interleaved list.

    Returns
    -------
    arglist : list of str or dict or ndarray
    SafeDictForComplexTypes : dict

    Examples
    ------
    >>> kwargs = dict(a=5, b=7.7, c='stegosaurus')
    >>> kwlist, _ = kwargs_to_arglist(**kwargs)
    >>> kwlist[0:2]
    ['--a', '5']
    '''
    keys = kwargs.keys()
    keys.sort(key=len)  # sorty by length, smallest to largest
    arglist = list()
    SafeDict = dict()
    for key in keys:
        val = kwargs[key]
        if isinstance(val, dict) or isinstance(val, np.ndarray):
            SafeDict[key] = val
        else:
            arglist.append('--' + key)
            arglist.append(str(val))
    return arglist, SafeDict


def makeParserWithDefaultsFromConfigFiles(ReqArgs, Moves):
    ''' Create parser filled with default settings from config files

    Only certain sections of the config files are included,
    selected by the provided RequiredArgs and Moves

    Returns
    -------
    parser : argparse.ArgumentParser
        Has all expected kwargs defined by config files.
    '''
    parser = argparse.ArgumentParser()
    configFiles = _getConfigFileDict(ReqArgs)
    for fpath, secName in configFiles.items():
        if secName is not None:
            secName = ReqArgs[secName]
        if fpath.count('moves') > 0:
            for moveName in Moves:
                fillParserWithDefaultsFromConfigFile(parser, fpath, moveName)
        else:
            fillParserWithDefaultsFromConfigFile(parser, fpath, secName)
    return parser


def _getConfigFileDict(ReqArgs):
    ''' Returns dict of config files to inspect for parsing keyword options,

    These files contain default settings for bnpy.

    Returns
    --------
    ConfigPathMap : dict with key/value pairs s.t.
    * key : absolute filepath to config file
    * value : string name of required arg

    Examples
    --------
    >> CMap = _getConfigFilePathMap()
    >> CMap[ '/path/to/bnpy/bnpy/config/allocmodel.conf' ]
    'allocModelName'
    '''
    bnpyroot = os.path.sep.join(
        os.path.abspath(__file__).split(os.path.sep)[:-2])
    cfgroot = os.path.join(bnpyroot, 'config/')
    ConfigPathMap = {
        cfgroot + 'allocmodel.conf': 'allocModelName',
        cfgroot + 'obsmodel.conf': 'obsModelName',
        cfgroot + 'learnalg.conf': 'algName',
        cfgroot + 'init.conf': None,
        cfgroot + 'output.conf': None,
        cfgroot + 'moves.conf': None,
    }
    OnlineDataConfigPath = cfgroot + 'onlinedata.conf'
    if ReqArgs['algName'] in OnlineDataAlgSet:
        ConfigPathMap[OnlineDataConfigPath] = None
    return ConfigPathMap


def fillParserWithDefaultsFromConfigFile(parser, confFile,
                                         targetSectionName=None):
    ''' Add default arg key/value pairs from confFile to the parser.

    Post Condition
    ------
    parser updated in-place with key/value pairs from config file.
    '''
    config = _readConfigFile(confFile)
    for curSecName in config.sections():
        if curSecName.count("Help") > 0:
            continue
        if targetSectionName is not None:
            if curSecName != targetSectionName:
                continue

        DefDict = dict(config.items(curSecName))
        try:
            HelpDict = dict(config.items(curSecName + "Help"))
        except ConfigParser.NoSectionError:
            HelpDict = dict()

        group = parser.add_argument_group(curSecName)
        for argName, defVal in DefDict.items():
            defType = _getTypeFromString(defVal)
            if argName in HelpDict:
                helpMsg = '[%s] %s' % (defVal, HelpDict[argName])
            else:
                helpMsg = '[%s]' % (defVal)
            argName = '--' + argName

            if defType == str:
                # Don't enforce types, so we can pass dicts
                group.add_argument(argName, default=defVal, help=helpMsg)
            else:
                group.add_argument(
                    argName, default=defVal, help=helpMsg, type=defType)


def _readConfigFile(filepath):
    ''' Read entire configuration from a .conf file
    '''
    config = ConfigParser.SafeConfigParser()
    config.optionxform = str
    config.read(filepath)
    return config


def _getTypeFromString(defVal):
    ''' Determine Python type from the provided default value.

    Returns
    ---------
    t : type
        type object. one of {int, float, str} if defVal is a string
        otherwise, evaluates and returns type(defVal)

    Examples
    ------
    >>> _getTypeFromString('deinonychus')
    <type 'str'>
    >>> _getTypeFromString('3.14')
    <type 'float'>
    >>> _getTypeFromString('555')
    <type 'int'>
    >>> _getTypeFromString('555.0')
    <type 'float'>
    >>> _getTypeFromString([1,2,3])
    <type 'list'>
    '''
    if not isinstance(defVal, str):
        return type(defVal)
    try:
        int(defVal)
        return int
    except Exception as e:
        pass
    try:
        float(defVal)
        return float
    except Exception as e:
        pass
    return str


def organizeParsedArgDictIntoSections(ReqArgs, Moves, kwargs):
    ''' Organize 'flat' dictionary of key/val pairs into sections

    Returns
    --------
    finalArgDict : dict
        with sections for algName, obsModelName, etc.
    '''
    outDict = dict()
    configFileDict = _getConfigFileDict(ReqArgs)
    for fpath, secName in configFileDict.items():
        if secName is not None:
            secName = ReqArgs[secName]
        if fpath.count('moves') > 0:
            for moveName in Moves:
                addArgsFromSectionToDict(kwargs, fpath, moveName, outDict)
        else:
            addArgsFromSectionToDict(kwargs, fpath, secName, outDict)

    return outDict


def addArgsFromSectionToDict(inDict, confFile, targetSecName, outDict):
    ''' Transfer 'flat' dictionary kwargs into argDict by section
    '''
    config = _readConfigFile(confFile)
    for secName in config.sections():
        if secName.count("Help") > 0:
            continue
        if targetSecName is not None:
            if secName != targetSecName:
                continue
        BigSecDict = dict(config.items(secName))
        secDict = dict([(k, v)
                        for (k, v) in inDict.items() if k in BigSecDict])
        outDict[secName] = secDict
    return outDict


def parse_task_ids(jobpath, taskids=None, dtype=str):
    ''' Get list of task ids as strings.

    Returns
    ---------
    taskidstrs : list of type dtype [default=str]
        each entry is string representation of taskid
        sorted in order from smallest to largest if convertible to int,
        otherwise sorted in alphabetical order.

    Examples
    ---------
    >>> parse_task_ids(None, taskids=[4,5,6.0])
    ['4', '5', '6']
    >>> parse_task_ids(None, taskids='1-3')
    ['1', '2', '3']
    >>> parse_task_ids(None, taskids='4')
    ['4']
    >>> parse_task_ids(None, taskids='4,11,5,2')
    ['2', '4', '5', '11']
    >>> parse_task_ids(None, taskids='.best,.worst')
    ['.best', '.worst']
    '''
    import glob
    import numpy as np

    if taskids is None or taskids == 'all':
        fulltaskpaths = glob.glob(os.path.join(jobpath, '*'))
        taskids = [os.path.split(tpath)[-1] for tpath in fulltaskpaths]
    elif isinstance(taskids, str):
        if taskids.count(',') > 0:
            taskids = taskids.split(',')
        elif taskids.count('-') > 0:
            fields = taskids.split('-')
            if not len(fields) == 2:
                raise ValueError("Bad taskids specification")
            fields = np.int32(np.asarray(fields))
            taskids = np.arange(fields[0], fields[1] + 1).tolist()
        elif taskids.startswith('.'):
            # Special case: '.worst' or '.best'
            return [taskids]
    # Convert to list of ints
    try:
        if not isinstance(taskids, list):
            taskids = [int(taskids)]
        else:
            taskids = [int(t) for t in taskids]
        taskids.sort()  # sort the integers (not strings)!
    except ValueError as e:
        if not str(e).count('int'):
            raise e
        taskids = sorted([str(t) for t in taskids])
    # Return list of strings
    taskidstrs = [dtype(t) for t in taskids]
    return taskidstrs
