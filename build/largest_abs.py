import sys
import numpy

filename = sys.argv[1]
vals = numpy.loadtxt(filename)
print "%s %i %.18e"%(filename, vals.shape[0], vals.max())
