from builtins import *
from libc.stdio cimport *
from libc.string cimport *
from libc.stdlib cimport atoi, atof, free

cdef extern from "stdio.h":
    int fclose(FILE*)
    FILE* fopen(char*, char*)
    ssize_t getline(char**, size_t*, FILE*)

def read_from_ldac_file(
        str filename, int N,
        int[:] dptr, int[:] wids, double[:] wcts):
    filename_byte_string = filename.encode("UTF-8")
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "r")
    if cfile == NULL:
        raise IOError("No such file or directory: '%s'" % filename)

    cdef char* line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    cdef int n = 0
    cdef int d = 1
    cdef int N_d = 0
    while True:
        # TODO: Avoid the realloc each time getline is called
        line = NULL
        read = getline(&line, &l, cfile)
        # Stop at END_OF_FILE
        if read == -1:
            break
        # Read the total number of tokens in this doc
        N_d = atoi(line)
        # Advance ptr past the next " " (ascii code 32)
        line += 1
        while line[0] != 32:
            line += 1
        line += 1
        # Repeat for each token id/count pair (N_d of these)
        for tpos in range(0, N_d):
            # Read the int id
            wids[n] = atoi(line)
            # Advance ptr past the ":" (ascii code 58)
            line += 1
            while line[0] != 58:
                line += 1
            line += 1
            # Read the count
            wcts[n] = atof(line)
            # Advance ptr past the " " (ascii code 32)
            if tpos < N_d - 1:
                line += 1
                while line[0] != 32:
                    line += 1
                line += 1
            # print wid, wct
            if n >= N:
                raise IndexError("Provided N too small. n=%d" % (n))
            n += 1
        if d >= N:
            raise IndexError("Provided N too small for docs. d=%d" % (d))
        dptr[d] = n
        d += 1

    free(line);
    fclose(cfile)
    return n, d
