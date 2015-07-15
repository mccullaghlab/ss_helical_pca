#!/Users/martinmccullagh/anaconda/bin/python
##!/Library/Frameworks/Python.framework/Versions/Current/bin/python
#USAGE : 
import scipy
import sys
import os
import numpy
import math
import MDAnalysis
from MDAnalysis.analysis.align import *

# constant to convert radians to degrees
radians_to_degrees = 180.0/3.1415926535
# number of types of bases (e.g. purines and pyrimidines)
n_base_types = 3

# defining which residues are each base type
res_select = []
res_select.append("resname DA or resname DA3 or resname DA5 or resname ADE or resname DG or resname DG3 or resname DG5 or resname GUA or resname AP3")
res_select.append("resname DT or resname DT3 or resname DT5 or resname THY or resname DC or resname DC3 or resname DC5 or resname CYT or resname DU or resname DU3 or resname DU5 or resname URA")
res_select.append("resname PDI")
# Atom selections for bases
base_atom_select = []
base_atom_select.append("name N9 or name N7 or name C8 or name C5 or name C4 or name N3 or name C2 or name N1 or name C6 or name O6 or name N2 or name N6 or name O2 or name N4 or name O4 or name C5M or name N2")
base_atom_select.append("name N9 or name N7 or name C8 or name C5 or name C4 or name N3 or name C2 or name N1 or name C6 or name O6 or name N2 or name N6 or name O2 or name N4 or name O4 or name C5M or name N2")
base_atom_select.append("name N1 or name C8 or name O1 or name C9 or name O2 or name C10 or name C11 or name C12 or name C13 or name C14 or name C15 or name C16 or name C17 or name C18 or name C19 or name C20 or name C21 or name C22 or name C23 or name C24 or name C25 or name C26 or name C27 or name O3 or name C28 or name C29 or name C30 or name O4 or name N2 or name C31")
# Three atoms used to define axes of bases
atom = []
atom.append([])
atom[0].append("N9")   # first atom making primary axis
atom[0].append("N1")   # second atom making primary axis
atom[0].append("N7")   # third atom in plane. Used to determine out of plane axis
atom.append([])
atom[1].append("C6")   # pyrimidine
atom[1].append("N3")   # pyrimidine
atom[1].append("C2")   # pyrimidine
atom.append([])
atom[2].append("N1")   # PDI
atom[2].append("N2")   # PDI
atom[2].append("C16")  # PDI


# Subroutines

# read the configuration file and populate the global variables
def ParseConfigFile(cfg_file):
	global top_file, traj_file, out_file
	f = open(cfg_file)
	for line in f:
		# first remove comments
		if '#' in line:
			line, comment = line.split('#',1)
		if '=' in line:
			option, value = line.split('=',1)
			option = option.strip()
			value = value.strip()
			# check value
			if option.lower()=='topfile':
				top_file = value
			elif option.lower()=='trajfile':
				traj_file = value
			elif option.lower()=='outfile':
				out_file = value
			else :
				print "Option:", option, " is not recognized"

def computePbcDist(r1,r2,box):
	dist = 0

	for j in range(0,3):
		temp = r1[j]-r2[j]
		if temp < -box[j]/2.0:
			temp += box[j]
		elif temp > box[j]/2.0:
			temp -= box[j]
		dist += temp*temp

	dist = math.sqrt(dist)
	return dist;

# subroutine to compute the three axes for each base
def computeBaseAxes(nucl):
	global res_select, atom, base_atom_select, n_base_types, res_atom_select
	n_residues = len(nucl.resids())

	axes = numpy.zeros((n_residues,3,3),dtype=float)

	for i in range(0,n_residues):

		# determine what type of base we are looking at
		sel1 = "resid " + str(nucl.resids()[i])
		sel1_univ = nucl.selectAtoms(sel1)
		for j in range(0,n_base_types):
			sel2_univ = sel1_univ.selectAtoms(res_select[j])
			if len(sel2_univ.atoms) > 0:
				atom1 = atom[j][0]
				atom2 = atom[j][1]
				atom3 = atom[j][2]
				res_atom_select.append(base_atom_select[j])
				break

		# select the residue of interest
		sel1 = "resid " + str(nucl.resids()[i]) + " and name "+atom1
		sel2 = "resid " + str(nucl.resids()[i]) + " and name "+atom2
		sel3 = "resid " + str(nucl.resids()[i]) + " and name "+atom3
		sel1_univ = nucl.selectAtoms(sel1)
		sel2_univ = nucl.selectAtoms(sel2)
		sel3_univ = nucl.selectAtoms(sel3)
		# get the three atom positions
		atom1_pos = sel1_univ.positions[0]
		atom2_pos = sel2_univ.positions[0]
		atom3_pos = sel3_univ.positions[0]
		# determine the three axes using the three atom positions
		r1 = atom2_pos-atom1_pos
		r1 /= math.sqrt(numpy.dot(r1,r1))
		t1 = atom3_pos-atom1_pos
		t1 /= math.sqrt(numpy.dot(t1,t1))
		r3 = numpy.cross(r1,t1)
		r3 /= math.sqrt(numpy.dot(r3,r3))
		r2 = numpy.cross(r3,r1)
		r2 /= math.sqrt(numpy.dot(r2,r2))
		# save the axes for each base
		axes[i][0] = r1
		axes[i][1] = r2
		axes[i][2] = r3

	return axes;

# subroutine to compute three distances along axis of first base for each pair of bases
def computeBaseDistances(nucl,base_axes,array):
	global res_atom_select
	n_residues = len(nucl.resids())
	coms = numpy.zeros((n_residues,3),dtype=float)

	for i in range(0,n_residues):
		# select the residue of interest
		selection = "resid " + str(nucl.resids()[i])
		res_univ = nucl.selectAtoms(selection)
		base_univ = res_univ.selectAtoms(res_atom_select[i])
		# get the center of mass of the base
		coms[i] = base_univ.centerOfMass()

	counter = 0
	for i in range(0,n_residues-1):
		for j in range (i+1,n_residues):
			dist = coms[j]-coms[i]
			slide = numpy.dot(dist,base_axes[i][0])
			shift = numpy.dot(dist,base_axes[i][1])
			rise = numpy.dot(dist,base_axes[i][2])
			array[counter][0] = rise
			array[counter][1] = shift
			array[counter][2] = slide
			counter += 1
#			dist_file_pointer.write("%3d-%3d %8.3f %8.3f %8.3f\n" % (i+1,j+1,slide,shift,rise))

# subroutine to compute the three angles needed to rotate base one to base two around the axes of base one for each pair of bases. 
def computeBaseAngles(nucl,base_axes,array):
	global radians_to_degrees
	n_residues = len(nucl.resids())
	
	counter = 0
	for i in range(0,n_residues-1):
		for j in range (i+1,n_residues):
			# Roll is the rotation about the major axis (r1) 
			# determine this by first projecting r2' into the r2xr3 plane
			r3p = numpy.dot(base_axes[j][2],base_axes[i][1])*base_axes[i][1] + numpy.dot(base_axes[j][2],base_axes[i][2])*base_axes[i][2]
			r3p /= math.sqrt(numpy.dot(r3p,r3p))
			cos_roll = numpy.dot(r3p,base_axes[i][2])
			sin_roll = numpy.dot(r3p,base_axes[i][1])
			roll = math.atan2(sin_roll,cos_roll)*radians_to_degrees
			# Tilt is the rotation about the in-plane minor axis (r2)
			# we project r3' into the r1xr3 plane
			r3p = numpy.dot(base_axes[j][2],base_axes[i][0])*base_axes[i][0]+numpy.dot(base_axes[j][2],base_axes[i][2])*base_axes[i][2]
			r3p /= math.sqrt(numpy.dot(r3p,r3p))
			cos_tilt = numpy.dot(r3p,base_axes[i][2])
			sin_tilt = numpy.dot(r3p,base_axes[i][0])
			tilt = math.atan2(sin_tilt,cos_tilt)*radians_to_degrees
			# Twist is the rotation about the out-of-plane axis (r3)
			# we project r1' into the r1xr2 plane
			r2p = numpy.dot(base_axes[j][1],base_axes[i][0])*base_axes[i][0]+numpy.dot(base_axes[j][1],base_axes[i][1])*base_axes[i][1]
			r2p /= math.sqrt(numpy.dot(r2p,r2p))
			cos_twist = numpy.dot(r2p,base_axes[i][1])
			sin_twist = numpy.dot(r2p,base_axes[i][0])
			twist = math.atan2(sin_twist,cos_twist)*radians_to_degrees
			array[counter][0] = twist
			array[counter][1] = tilt
			array[counter][2] = roll
			counter += 1
			# print values to file for this pair of bases
			#ang_file_pointer.write("%3d-%3d %8.3f %8.3f %8.3f\n" % (i+1,j+1,roll,tilt,twist))

def computePCA(data):

	covar_mat = numpy.dot(data.T,data)

	u,s,v = numpy.linalg.svd(covar_mat)

	return v

# Main Program

# read in command line argument
cfg_file = sys.argv[1]

# read cfg file
ParseConfigFile(cfg_file)

# initiate coordinate universe
coord = MDAnalysis.Universe(top_file, traj_file)

# determine total selection
nucleic_res_select = res_select[0]
for i in range(1,n_base_types):
	nucleic_res_select += " or "+ res_select[i]

# Select only nucleic atoms
nucl = coord.selectAtoms(nucleic_res_select)
print nucl


n_bases = len(nucl.resids())
n_bps = n_bases*(n_bases-1)/2
# declare data arrays
trans_coord = numpy.empty((n_bps*coord.trajectory.numframes,3),dtype=float)
ang_coord = numpy.empty((n_bps*coord.trajectory.numframes,3),dtype=float)

# write log information
print "Topology file:", top_file
print "Trajectory file:", traj_file
print "Number of frames in traj file:", coord.trajectory.numframes
print "Number of bases:", n_bases

# Loop through trajectory
res_atom_select=[]
array_start = 0
for ts in coord.trajectory:


	# compute axes of base
	base_axes = computeBaseAxes(nucl)

	# compute and print distance parameters for each pair of bases
	computeBaseDistances(nucl,base_axes,trans_coord[array_start:array_start+n_bps])

	# compute and print angle parameters for each pair of bases
	computeBaseAngles(nucl,base_axes,ang_coord[array_start:array_start+n_bps])

	# move up array start
	array_start += n_bps

trans_pca = computePCA(trans_coord)
rot_pca = computePCA(ang_coord)

# open output files
out = open(out_file,'w')

for i in range(n_bps*coord.trajectory.numframes):

	trans = numpy.dot(trans_coord[i],trans_pca[0])
	rot = numpy.dot(ang_coord[i],rot_pca[0])
	out.write("%10.5f %10.5f\n" % (trans, rot))


out.close

