
EIGSDIR ?= ../data
DEGSDIR ?= ../data

plfitfiles := $(shell ls ../src/*.c | grep -F -v plgen.c)

all: plfit plfiles

plfit: $(plfitfiles)
	gcc -o plfit -I../src $(plfitfiles) -lm

#degsfiles := $(wildcard **.degs)
#eigsfiles := $(wildcard **.eigs)

degsfiles := $(shell find $(DEGSDIR) -type f -name '*.degs')
eigsfiles := $(shell find $(EIGSDIR) -type f -name '*.eigs')

plfiles: $(degsfiles:.degs=.plfit) $(eigsfiles:.eigs=.plfit)

# all the targets are the degree files converted into plfit files


# Convert .degs files into .plfit files 
# $< is the name of the degree file
# $@ is the name of the plfit file.
%.plfit : %.degs
	tail -n +2 $< | ./plfit -M -p exact > $@ 
    
%.plfit : %.eigs
	sed 's/^-//' $< | ./plfit -M -p exact > $@

