import sys
import numpy

filename = sys.argv[1]
vals = numpy.loadtxt(filename)
n = vals.shape[0]
minv = n*numpy.finfo(float).eps
absvals = numpy.abs(vals)
absvals = absvals[absvals >= minv]
numpy.savetxt(sys.stdout, absvals, fmt="%.18e")

