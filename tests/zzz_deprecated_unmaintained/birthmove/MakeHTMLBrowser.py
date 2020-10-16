import glob
import argparse
import os
import sys
import numpy as np

global OUTPUTDIR
OUTPUTDIR = "/tmp/"
CELL_WIDTH = 200

htmlstart = """
    <html>
    <style>
    td {
        border: 0px solid black;
        width: %dpx;
        text-align: center;
        padding-bottom: 10px;
        padding-left: 10px;
    }
    </style>
    <body>
    <div align=center>
    """ % (CELL_WIDTH)

htmlend = """
    </div>
    </body>
    </html>
    """

def makeSingleRunHTMLNavBar(
        rowVarName='', rowVal='', colVarName='', colVal='',
        rowVals=[], colVals=[],
        **kwargs):
    return "<h2> %s=%s &nbsp;&nbsp;&nbsp; %s=%s </h2>" % (
        rowVarName, rowVal, colVarName, colVal)

def makeImgTag(varname1, val1, varname2, val2, imgprefix="ELBOGain"):
    flist = sorted(glob.glob("%s/%s*.png" % (OUTPUTDIR, imgprefix)))    
    str1 = "%s=%s" % (varname1, val1)
    str2 = "%s=%s" % (varname2, val2)
    for fname in flist:
        basename = fname.split(os.path.sep)[-1]
        if basename.count(str1) and basename.count(str2):
            htmltag = "<img src=%s width=%d>" % (basename, CELL_WIDTH)
            return htmltag
    #for f in flist:
    #    print f
    print("Cannot find file: %s %s" % (str1, str2))
    return ''


def makeSingleRunHTMLStr(
        rowVarName, rowVal, colVarName, colVal,
        rowVals=[], colVals=[]):
    '''
    '''
    rellinkfilename = "RunInfo_%s=%s_%s=%s.html" % (
        rowVarName, rowVal, colVarName, colVal)

    htmlstr = htmlstart
    htmlstr += makeSingleRunHTMLNavBar(**locals())
    htmlstr += "<table>"    
    htmlstr += "<tr>"
    htmlstr += "<td>Complete model before proposal.</td>"
    htmlstr += "<td>%s</td>" % (
        makeImgTag(rowVarName, rowVal, colVarName, colVal, "BeforeComps"))
    htmlstr += "</tr>\n"

    htmlstr += "<tr>"
    htmlstr += "<td>Docs used for initial proposal." + \
               "Showing at most 25 docs " + \
               "selected at random from batch 1.</td>"
    htmlstr += "<td>%s</td>" % (
        makeImgTag(rowVarName, rowVal, colVarName, colVal, "FirstDocs"))
    htmlstr += "</tr>\n"

    htmlstr += "<tr>"
    htmlstr += "<td>Soft assignments of tokens in those docs to target topic.</td>"
    htmlstr += "<td>%s</td>" % (
        makeImgTag(rowVarName, rowVal, colVarName, colVal, "FirstRelevantDocs"))
    htmlstr += "</tr>\n"

    htmlstr += "<tr>"
    htmlstr += "<td>Brand new clusters after initial proposal," + \
               "using only data from batch 1</td>"
    htmlstr += "<td>%s</td>" % (
        makeImgTag(rowVarName, rowVal, colVarName, colVal, "FirstFreshComps"))
    htmlstr += "</tr>\n"

    htmlstr += "<tr>"
    htmlstr += "<td>Complete model after proposal." + \
               "Informed by ALL batches</td>"
    htmlstr += "<td>%s</td>" % (
        makeImgTag(rowVarName, rowVal, colVarName, colVal, "AfterComps"))
    htmlstr += "</tr>\n"

    htmlstr += "<tr>"
    htmlstr += "<td>ELBO objective improvement</td>"
    htmlstr += "<td>%s</td>" % (
        makeImgTag(rowVarName, rowVal, colVarName, colVal, "ELBOgain"))
    htmlstr += "</tr>\n"
    htmlstr += "</table>"    
    htmlstr += htmlend
    return rellinkfilename, htmlstr


def makeMainHTMLStr(
        rowVarName, rowVals,
        colVarName, colVals,
        mainpagefilename='index.html'):
    '''
    '''
    htmlstr = htmlstart
    htmlstr += "<table>"

    # Header row
    htmlstr += "<tr>"
    htmlstr += "<td></td>"
    for colid, colval in enumerate(colVals):
        htmlstr += "<td>%s=%s</td>" % (
            colVarName, colval)
    htmlstr += "</tr>\n"

    # Sneakpeak data row
    htmlstr += "<tr>"
    htmlstr += "<td>Original model, before birth</td>"
    for colid, colval in enumerate(colVals):
        imgfiletag = makeImgTag(
            rowVarName, rowVals[-1], colVarName, colval, "BeforeComps")
        htmlstr += "<td>%s</td>" % (imgfiletag)
    htmlstr += "</tr>\n"

    # Content rows
    for rowid, rowval in enumerate(rowVals):
        htmlstr += "<tr>"
        htmlstr += "<td>%s=%s</td>" % (rowVarName, rowval)

        for colid, colval in enumerate(colVals):
            # Make page for this cell
            subpagerellink, subpagehtmlstr = makeSingleRunHTMLStr(
                rowVarName, rowval, colVarName, colval, rowVals, colVals)

            with open(os.path.join(OUTPUTDIR,subpagerellink), 'w') as f:
                f.write(subpagehtmlstr)

            imgfiletag = makeImgTag(
                rowVarName, rowval, colVarName, colval, "ELBOgain")
            htmlstr += "<td><a href=%s>%s</a></td>" % (
                subpagerellink, imgfiletag)
        htmlstr += "</tr>\n"

    htmlstr += "</table>"
    htmlstr += htmlend
    with open(os.path.join(OUTPUTDIR, mainpagefilename), 'w') as f:
        f.write(htmlstr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outputdir")
    args = parser.parse_args()

    OUTPUTDIR = args.outputdir

    rowVarName = "nDocPerBatch"
    rowVals = ["4", "16", "64", "256"]
    colVarName = "nWordsPerDoc"
    colVals = ["64", "128", "256", "512"]
    makeMainHTMLStr(rowVarName, rowVals, colVarName, colVals)
