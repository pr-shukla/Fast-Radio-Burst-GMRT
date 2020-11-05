import numpy as Num
import matplotlib.pyplot as plt
import bisect, os, sys, getopt, glob
import scipy, scipy.signal, scipy.stats
from optparse import OptionParser
import time

start = time.time()

class candidate:
	def __init__(self, DM, sigma, ime, bin, downfact):
		self.DM = DM
		self.ime = ime
		self.sigma = sigma
		self.bin = bin
		self.downfact = downfact
	def __str__(self):
		return "%7.2f %7.2f %13.6f %10d   %3d\n"%\
			(self.DM, self.sigma, self.ime, self.bin, self.downfact)
	def __cmp__(self, other):
		# Sort by ime (i.e bin) by default
		return cmp(self.bin, other.bin)

def cmp_sigma(self, other):
	# Comparison functio to sort candidates by significance
	retval = -cmp(self.sigma, other.sigma)
	return retval

def fft_convolve(fftd_data, fftd_kern, lo, hi):
    # Note:  The initial FFTs should be done like:
    # fftd_kern = rfft(kernel, -1)
    # fftd_data = rfft(data, -1)
	prod = Num.multiply(fftd_data, fftd_kern)
	prod.real[0] = fftd_kern.real[0] * fftd_data.real[0]
	prod.imag[0] = fftd_kern.imag[0] * fftd_data.imag[0]
	return Num.fft.rfft(prod, 1).astype(Num.float32)

def make_fftd_kerns(downfacts, fftlen):
	fftd_kerns = []
	for downfact in downfacts:
		kern = Num.zeros(fftlen, dtype = Num.float32)
        # These offsets produce kernels that give results
        # equal to scipy.signal.convolve
		if downfact % 2:  # Odd number
			kern[:int(downfact/2+1)] += 1.0
			kern[int(-(downfact/2)):] += 1.0
		else:             # Even number
			kern[:int(downfact/2+1)] += 1.0
			if (downfact > 2):
				kern[-int((downfact/2-1)):] += 1.0
         # The following normalization preserves the
        # RMS=1 characteristic of the data
		fftd_kerns.append(Num.fft.rfft(kern / Num.sqrt(downfact), axis = -1))
	return fftd_kerns

def prune_related1(hibins, hivals, downfact):
    # Remove candidates that are close to other candidates
    # but less significant.  This one works on the raw 
    # candidate arrays and uses the single downfact
    # that they were selected with.
	toremove = set()
	for ii in range(0, len(hibins)-1):
		if ii in toremove: continue
		xbin, xsigma = hibins[ii], hivals[ii]
		for jj in range(ii+1, len(hibins)):
			ybin, ysigma = hibins[jj], hivals[jj]
			if (abs(ybin-xbin) > downfact/2):
				break
			else:
				if jj in toremove:
					continue
				if (xsigma > ysigma):
					toremove.add(jj)
				else:
					toremove.add(ii)
                    # Now zap them starting from the end
	toremove = sorted(toremove, reverse = True)
	bdlist = []
	for bin in toremove:
		bdlist.append(hibins[bin])
		del(hibins[bin])
		del(hivals[bin])
	return hibins, hivals, bdlist

def prune_related2(dm_candlist, downfacts):
    # Remove candidates that are close to other candidates
    # but less significant.  This one works on the candidate 
    # instances and looks at the different downfacts of the
    # the different candidates.
	toremove = set()
	for ii in xrange(0, len(dm_candlist)-1):
		if ii in toremove: continue
		xx = dm_candlist[ii]
		xbin , xsigma = xx.bin, xx.sigma
		for jj in xrange(ii+1, len(dm_candlist)):
			yy = dm_candlist[jj]
			ybin, ysigma = yy.bin, yy.sigma
			if (abs(ybin-xbin) > max(downfacts)/2):
				break
			else:
				if jj in toremove:
					continue
				prox = max([xx.downfact/2, yy.downfact/2, 1])
				if (abs(ybin-xbin) <= prox):
					if (xsigma > ysigma):
						toremove.add(jj)
					else:
						toremove.add(ii)
                        # Now zap them starting from the end
	toremove = sorted(toremove, reverse = True)
	bdlist = []
	for bin in toremove:
		bdlist.append(dm_candlist[bin])
		del(dm_candlist[bin])
	return dm_candlist, bdlist

def prune_border_cases(dm_candlist, offregions):
    # Ignore those that are located within a half-width
    # of the boundary between data and padding
    #print offregions
	toremove = set()
	for ii in range(len(dm_candlist)-1, -1, -1):
		cand = dm_candlist[ii]
		loside = cand.bin-cand.downfact/2
		hiside = cand.bin+cand.downfact/2
		if hiside < offregions[0][0]: break
		for off, on in offregions:
			if (hiside > off and loside < on):
				toremove.add(ii)
                # Now zap them starting from the end
	toremove = sorted(toremove, reverse = True)
	for ii in toremove:
		del(dm_candlist[ii])
	return dm_candlist

		

usage = "usage: %prog [options] .dat files _or_ .singlepulse files"

def read_singlepulse_files(infiles, threshold, T_start, T_end, lt, ut, ld, ud, ls, us, lw, uw, sm, dt):
	DMs = [] # Will store Different DM values
	candlist = [] # Will store Different SNR values
	ime = [] # Will store Different ime values
	width = [] # Will store Different width values
	noofpulse = [] # Will store Different number of pulse values
	c = [] # Will store Different SNR values for finding median of SNR
	for ii, infile in enumerate(infiles):
		if infile.endswith(".singlepulse") or infile.endswith(".ascii"):
			pass
		else:
			continue
		if os.stat(infile)[6]:
			try:
				cands = Num.loadtxt(infile)
				if (len(cands.shape) == 1):
					c.append(cands[1])
				else:
					for x in cands:
						
						c.append(x[1])
			except: # No candidate in the file
				IndexError
	
	c.sort()
	smedian = c[int(len(c)/2)] # finding value of median of SNR
	
	if sm:
		ls = smedian

	for ii, infile in enumerate(infiles):
		if infile.endswith(".singlepulse") or infile.endswith(".ascii"):
			pass
		else:
			continue
		if os.stat(infile)[6]:
			try:
				cands = Num.loadtxt(infile)
				if (len(cands.shape) == 1 and ld <= cands[0] <= ud and lt <=cands[2] <= ut and ls <= cands[1] <= us and lw <= cands[4]*dt <= uw):
					DMs.append(cands[0])
					candlist.append(cands[1])
					ime.append(cands[2])
					width.append(cands[4]*dt)
					noofpulse.append(int(1))
				
				else:
					for x in cands:
						if (ld <= x[0] <= ud and lt <=x[2] <= ut and ls <= x[1] <= us and lw <= x[4]*dt <= uw):
							DMs.append(x[0])
							candlist.append(x[1])
							ime.append(x[2])
							width.append(x[4]*dt)
							noofpulse.append(cands.shape[0])
			except: # No candidate in the file
				IndexError
	
	return DMs, candlist, ime, width, noofpulse

usage = "usage: %prog [options] .dat files _or_ .singlepulse files"

parser = OptionParser(usage)

parser.add_option("-x", "--xwin", action="store_true", dest="xwin",
		default=False, help="Don't make a postscript plot, just use an X-window")
parser.add_option("-p", "--noplot", action="store_false", dest="makeplot",
		default=True, help="Look for pulses but do not generate a plot")
parser.add_option("-m", "--maxwidth", type="float", dest="maxwidth", default=0.0,
		help="Set the max downsampling in sec (see below for default)")
parser.add_option("-t", "--threshold", type="float", dest="threshold", default=5.0,
		help="Set a different threshold SNR (default=5.0)")
parser.add_option("-s", "--start", type="float", dest="T_start", default=0.0,
		help="Only plot events occuring after this ime (s)")
parser.add_option("-e", "--end", type="float", dest="T_end", default=1e9,
		help="Only plot events occuring before this ime (s)")
parser.add_option("-g", "--glob", type="string", dest="globexp", default=None,
		help="Process the files from this glob expression")
parser.add_option("-f", "--fast", action="store_true", dest="fast",
		default=False, help="Use a faster method of de-trending (2x speedup)")
parser.add_option("-b", "--nobadblocks", action="store_false", dest="badblocks",
		default=True, help="Don't check for bad-blocks (may save strong pulses)")
parser.add_option("-d", "--detrendlen", type="int", dest="detrendfact", default=1,
		help="Chunksize for detrending (pow-of-2 in 1000s)")
parser.add_option("-l", "--tsample", type="float", dest="imesample", default= 0.16384,
		help="Interval of each ime bin in millisecond")
parser.add_option("-n", "--nfile", type="string", dest="filename",
		help="Name of the file")
parser.add_option("--bd", "--badfile", type="string", dest="debug",
		help="Writes files with bad data")
parser.add_option("--lt", "--lowerime", dest="lime", type="float",
		default= 0.0, help="Lower Limit of the ime")
parser.add_option("--ut", "--upperime", dest="uime", type="float",
		default= 10000.0, help="Upper Limit of the ime")
parser.add_option("--ld", "--lowerdm", dest="ldm", type="float",
		default= 0.0, help="Lower Limit of the DM")
parser.add_option("--ud", "--upperdm", dest="udm", type="float",
		default= 50000.0, help="Upper Limit of the DM")
parser.add_option("--ls", "--lowersnr", dest="lsnr", type="float",
		default= 0.0, help="Lower Limit of the SNR")
parser.add_option("--us", "--uppersnr", dest="usnr", type="float",
		default= 50000.0, help="Upper Limit of the SNR")
parser.add_option("--lw", "--lowerwidth", dest="lwidth", type="float",
		default= 0.0, help="Lower Limit of the Width in millisecond")
parser.add_option("--uw", "--upperwidth", dest="uwidth", type="float",
		default= 50000.0, help="Upper Limit of the Width in millisecond")
parser.add_option("--sm", "--smedian", dest="snrmedian", action = "store_true",
		default= False , help="Whether lower limit of snr is median or not")
parser.add_option("--sk", "--skip", dest="askip", type = "string",
		default= "False" , help="Whether to skip association or not")
parser.add_option("--ef", "--expfactor", dest="efactor", type="float",
		default= 5.0 , help="Exponential factor for variation in size of symbols as snr") # (addition 0)
parser.add_option("--cs", "--colorsaturation", dest="csaturation", type="float",
		default= 50.0 , help="Value of SNR after which color of symbols will not change") # (addition 0)


(opts, args) = parser.parse_args()


if opts.askip == "True":
	sd = 1
else:
	sd = 0

if sd == 0:
	a = 0
	
	lt = opts.lime
	ut = opts.uime
	ld = opts.ldm
	ud = opts.udm
	ls = opts.lsnr
	us = opts.usnr
	lw = opts.lwidth
	uw = opts.uwidth
	sm = opts.snrmedian
	dt = opts.imesample
	ef = opts.efactor # (addition 0) 
	cs = opts.csaturation # (addition 0)

	useffts = True
	dosearch = True
	fftlen = 8192
	chunklen = 8000
	assert(opts.detrendfact in [1,2,4,8,16,32])
	detrendlen = int(opts.detrendfact*1000)
	if (detrendlen > chunklen):
		chunklen = detrendlen
		fftlen = int(next2_to_n(chunklen))
	blocks_per_chunk = chunklen / detrendlen
	overlap = int((fftlen - chunklen)/2)
	worklen = int(chunklen + 2*overlap)

	max_downfact = 30
	default_downfacts = [2,3,4,6,9,14,20,30,45,70,100,150,220,300] # For convolution of SNR for  particular DM

	if args[0].endswith(".singlepulse"):  # For plotting
		filenmbase = args[0][:args[0].rfind(".singlepulse")]
		dosearch = False
	elif args[0].endswith(".dat"): # For ime association  
		filenmbase = args[0][:args[0].rfind(".dat")]
	else:
		filesnmbase = args[0]


	if not dosearch:		##Corerection20190415 DM replaced by DM_ similarly for candlist and others	
		DM_, candlist_, ime_, width_, noofpulse_ = \
                           read_singlepulse_files(args, opts.threshold, opts.T_start, opts.T_end, lt, ut, ld, ud, ls, us, lw, uw, sm, dt) 


	else:
		DM_ = []
		candlist_ = []
		ime_ = []
		width_ = []
		noofpulse_ = []
		c = []
		f = Num.fromfile(args[0], dtype = Num.float32) # Stores SNR as an Array
		l = len(f)
		DM =[] # Will store DM value from Raw data
		candlist = []  # Will store SNR value from Raw data
		ime =[]  # Will store ime value from Raw data
		SNR = []  # Will store DM value from Raw data
		width = []  # Will store Width value from Raw data
		num_v_DMstr = {}
		for i in range(0,l,4):
			DM.append(f[i])
			ime.append(f[i+1])
			SNR.append(f[i+2])
			width.append(f[i+3])
		DM_sorted = [] # Will store DM in sorted form without repeatation in values of DM
		for i in DM:
			if i not in DM_sorted:
				DM_sorted.append(i)

		DM_sorted.sort()
		tz = []
		dz = []
		snrz = []
		wz = []
		T = ime
		D = DM
		for i in range(len(DM_sorted)):
			tz.append([])
			dz.append([])
			snrz.append([])
			wz.append([])

		for ii, i in enumerate(DM_sorted):
			for j,k in enumerate(DM):
				if i == k:
					tz[ii].append(ime[j])
					dz[ii].append(DM[j])
					snrz[ii].append(SNR[j])
					wz[ii].append(width[j])
		

		for i, ii in enumerate(DM_sorted):
			if len(snrz[i]) >= 1000: # If length of List of SNR corresponding to a particular DM is greater than 1000
				sort_imeseries = sorted(zip(tz[i],dz[i],snrz[i],wz[i]))
				t = []
				d = []
				snr = []
				w = []
				for v in sort_imeseries:
					t.append(v[0])
					d.append(v[1])
					snr.append(v[2])
					w.append(v[3])
				ind = [] 
				DMstr = "%.2f"%ii
				N, dt = len(snr), opts.imesample
				n_last = len(snr)%1000
				nn_last = len(snr)%100
				if len(snr)%1000 == 0:
					n_last = 1000
				if len(snr)%100 == 0:
					nn_last = 100
				ind = []
				DMstr = "%.2f"%ii
				N, dt = len(snr), opts.imesample
				roundN = N
				detrendlen = 100 # No. of items in a piece to be detrended
				chunklen = 1000 # No. of items in a piece that will undergo association
				overlap = 0
				obsime = N*dt
				if opts.maxwidth > 0.0:
					downfacts = [x for x in default_downfacts if x*dt <= opts.maxwidth]
				else:
					downfacts = [x for x in default_downfacts if x <= max_downfact] # storing reduced form of list 
                                                                                # default_downfact in given list upto 30
				if len(downfacts) == 0:
					downfacts = [default_downfacts[0]]
				if (i == 0):
					orig_N = N
					orig_dt = dt
				if useffts:
					fftd_kerns = make_fftd_kerns(default_downfacts, chunklen) # for the processing of first file
					fftd_kerns1 = make_fftd_kerns(default_downfacts, n_last) # for the processing of first file this
                                                                                    # list is created for last part of data
                                                                                    # which is obtained as reminder after 
                                                                                    # dividing this whole list by 1000



				

				numchunks = int(roundN / chunklen) + 1 # will be used later in processing when imeseries will be iterated with length
				if n_last == 1000:
					numchunks = int(roundN / chunklen)
                                                    # 1000 of each chunk
             # Split the imeseries into chunks for detrending
				numblocks = int(roundN / detrendlen)  # numblocks  = no. of rows in newly created imeseries below

				imeseries = Num.asarray(snr) # imeseries is list of SNR values for particular DM
				wd = Num.asarray(w) # Values in this list will be later stored as width
				td = Num.asarray(t) # Values in this list will be later stored as width
				eimeseries = imeseries[-(nn_last):] # End part of imeseries that is remained after division
                                                            # from 100
				imeseries = imeseries[:numblocks*100] # Whole imeseries after removing eimeseries 
				if nn_last == 100:
					imeseries = imeseries[:(numblocks-1)*100]
					imeseries.shape = (numblocks - 1, detrendlen)
					stds = Num.zeros(numblocks, dtype = Num.float64)
				else:
					imeseries.shape = (numblocks, detrendlen) # imeseries is reshaped 
                                                        # i.e each subpart is of length 1000 bins
					stds = Num.zeros(numblocks + 1, dtype = Num.float64)
				imeseries = list(imeseries) # imeseries is converted to type "list" for further operations
				imeseries.append(eimeseries) # Attaching Eimeseries to end of this newly reshaped imeseries
				
				
				ts = imeseries
				for kk, chunk in enumerate(imeseries): # iterating over each sub part of imeseries with variable chunk
					if opts.fast: # use median removal instead of detrending (2x speedup)
						tmpchunk = chunk.copy() # copying exactly the chunk in this list
						tmpchunk.sort() # sorting tmpchunk
						med = tmpchunk[len(tmpchunk)/2] # sorting out tmpchunk
						chunk -= med # subtracting median from each item of chunk      ----\ much faster method
						tmpchunk -= med # subtracting median from each item of tmpchunk----/ detrending data
					else:
                     # The detrend calls are the most expensive in the program
						ts[kk] = scipy.signal.detrend(chunk, type = 'linear')# detrending data using scipy.signal
						tmpchunk = ts[kk].copy() # copying that particular detrended part of imeseries in tmpchunk
						tmpchunk.sort() # sorting tmpchunk
                # The following gets rid of (hopefully) most of the 
                # outlying values (i.e. power dropouts and single pulses)
                # If you throw out 5% (2.5% at bottom and 2.5% at top)
                # of random gaussian deviates, the measured stdev is ~0.871
                # of the true stdev.  Thus the 1.0/0.871=1.148 correction below.
                # The following is roughly .std() since we already removed the median

					stds[kk] = Num.sqrt((Num.asarray(tmpchunk)**2.0).sum() / (0.95*len(tmpchunk))) # finding out standard deviation of each subpart of imesries and 
                                                       # storing that value in stds list by repplacing zeros inside it.
				stds *= 1.148 # multiplying each elements of std by 1.148 to remove error due to exclusion of end values of 
                          # each chunk
            # sort the standard deviations and separate those with
            # very low or very high values

				sort_stds = stds.copy() # copying exact stds list in the sort_stds list
				sort_stds.sort() # sorting the list sort_stds
            # identify the differences with the larges values (this
            # will split off the chunks with very low and very high stds


				locut = (sort_stds[1:int(numblocks/2)+1] - sort_stds[:int(numblocks/2)]).argmax() + 1# storing value of index in first half of imeseries with max. 
                                                           # difference (index of first list(sort_stds[1:numblocks/2+1]) 
                                                           # is stored)
				hicut = (sort_stds[int(numblocks/2)+1:] - sort_stds[int(numblocks/2):-1]).argmax() + int(numblocks/2) - 2# storing value of index in first half of 
                                                                           # imeseries with max. difference (index  
                                                                           # of first list(sort_stds[1:numblocks/2+1]) 
                                                                           # is stored)
				std_stds = scipy.std(sort_stds[locut:hicut])# stdandard deviation of list sort_std with its value taken from 
                                                         # locut to hicut
				median_stds = sort_stds[int((locut+hicut)/2)] # median of this list sort_stds is obtained


				if (opts.badblocks): # it is for removing the bad part of the imeseries whcich is not of our use
					lo_std = median_stds - 4.0 * std_stds # lower restriction for values in the list imeseries
					hi_std = median_stds + 44.0 * std_stds # upper restriction for values in the list imeseries
                    # Determine a list of "bad" chunks.  We will not search these.

					bad_blocks = Num.nonzero((stds < lo_std) | (stds > hi_std))[0] # storing indexes of items in stds list 
                                                                                # which are beyond restriction.
					stds[bad_blocks] = median_stds # replacing values of items in list stds with indexes stored in bad-blocks
                                                # by median_stds i.e. stds[with index in bad_blocks] = median_std
				else:
					bad_blocks = []

				bbb = []
				for r, rr in enumerate(imeseries):
					for s in imeseries[r]:
						bbb.append(s)
				imeseries = Num.asarray(bbb)	#again converting imeseries in the 1-D array
            # And set the data in the bad blocks to zeros
            # Even though we don't search these parts, it is important
            # because of the overlaps for the convolutions
	

				for bad_block in bad_blocks:
					loind, hiind = bad_block*detrendlen, bad_block*detrendlen + len(ts[bad_block])# storing starting index of item in 
                                                                              # imeseries which is lying in bad region
                                                                              # and storing last index of item in 
                                                                              # imeseries lying in bad region i.e we 
                                                                              # storing start and end index of bad subpart
                                                                              # of imeseries which was intially in the form
                                                                              # of chunk.
					imeseries[loind:hiind] = 0.0 # replacing bad part of data in imeseries by 0.0
					wd[loind:hiind] = 0.0
					td[loind:hiind] = 0.0
				 # Convert to a set for faster lookups below
				
				bad_blocks = set(bad_blocks)
                # Step through the data
				dm_candlist = [] # in futher part of programme we will be storing data of class candidate in this list
				for chunknum in range(int(numchunks)): # now we will iterate on imeseries in subparts of size 1000 with chunknum
                                                    # showing index of each subpart
					loind = chunknum*chunklen-overlap # lower index from where we will start copying values of imeseries in
                                                    # chunk
					hiind = (chunknum+1)*chunklen+overlap  # upper index upto which we will copying values of imeseires in chunk 
                # Take care of beginning and end of file overlap issues
					if chunknum == numchunks - 1:
						hiind = chunknum*chunklen + n_last + overlap# upper index (while dealing with end part of file) 
                        # upto which we will copying values of imeseires in chunk 
                # Take care of beginning and end of file overlap issues 
					if (chunknum==0):  # Beginning of file
						worklen = chunklen
						chunk = Num.zeros(worklen, dtype = Num.float32)  # storing 1000 zeros in in array chunk
						wchunk = Num.zeros(worklen, dtype = Num.float32)  # storing 1000 zeros in in array chunk
						tchunk = Num.zeros(worklen, dtype = Num.float32)  # storing 1000 zeros in in array chunk
						chunk[:] = imeseries[int(loind):int(hiind)]# replacing  values in chunk by 
                                                                #  items of imeseries
						wchunk[:] = Num.asarray(wd[int(loind):int(hiind)]) # replacing  values in chunk by 
                                                                        #  items of widths
						tchunk[:] = Num.asarray(td[int(loind):int(hiind)])# replacing  values in chunk by 
                                                                       #  items of ime
					elif (chunknum==numchunks-1):  # end of the imeseries
						worklen = n_last  # Number of items remaining in the last of imeseries after divsion from 1000
						chunk = Num.zeros(worklen, dtype = Num.float32) # storing "worklen" number of zeros in in array chunk
						wchunk = Num.zeros(worklen, dtype = Num.float32) # storing "worklen" number of zeros in in array chunk
						tchunk = Num.zeros(worklen, dtype = Num.float32) # storing "worklen" number of zeros in in array chunk
						chunk[:] = imeseries[int(loind):int(hiind)] # replacing  values in chunk by 
                                                                 #  items of imeseries
						wchunk[:] = Num.asarray(wd[int(loind):int(hiind)]) # replacing  values in chunk by 
                                                                        #  items of widths
						tchunk[:] = Num.asarray(td[int(loind):int(hiind)]) # replacing  values in chunk by 
                                                                       #  items of ime
					else:
						chunk = imeseries[int(loind):int(hiind)] # creating list 'chunk' by copying values in it from imeseries from 
                                                    # its index loind to hind
						wchunk = Num.asarray(wd[int(loind):int(hiind)])
						tchunk = Num.asarray(wd[int(loind):int(hiind)])
					# Make a set with the current block numbers					
					blocks_per_chunk = int(chunklen/detrendlen) # It gives number of stds list in each chunk with size 1000 i.e.
                                                            # we have 1000/100 = 10 std list in each chunk


					lowblock = blocks_per_chunk * chunknum # storing index of first item in list std which is present in chunk
					currentblocks = set(Num.arange(blocks_per_chunk) + lowblock) # storing index of all items in stds which are 
                                                                             # at presently inside chunk
					if chunknum == numchunks - 1: # When dealing with the end of the series
						currentblocks = set(Num.arange(int((n_last)/detrendlen))+lowblock)
					localgoodblocks = Num.asarray(list(currentblocks - bad_blocks)) - lowblock# removing all index from currentblocks which 
                                                                           # are also in bad_blocks then subracting lowblocks
                                                                           # from this array to get actual index of good
                                                                           # elements in chunk 
                 # Search this chunk if it is not all bad

					if len(localgoodblocks):
                     # This is the good part of the data (end effects removed)
						goodchunk = chunk # removing end parts of the data
                    # need to pass blocks/chunklen, localgoodblocks
                    # dm_candlist, dt, opts.threshold to cython routine

                    # Search non-downsampled data first
                    # NOTE:  these nonzero() calls are some of the most
                    #        expensive calls in the program.  Best bet would 
                    #        probably be to simply iterate over the goodchunk
                    #        in C and append to the candlist there.
						goodchunk = chunk
						hibins = Num.flatnonzero(goodchunk>opts.threshold) # storing index in hibins of items in goodchunks  
                                                                        # which are having values greater then threshold
						lobins = Num.flatnonzero(goodchunk<=opts.threshold)# storing index in lobins of items in goodchunks  
                                                                        # which are having values smaller then threshold
						lobins += chunknum * chunklen
						for sk in lobins:
							ind.append(sk)
						hivals = goodchunk[hibins] # storing values of items in goodchunk with indexes stored in hibins
						hiwds = wchunk[hibins]
						hitds = tchunk[hibins]
						hibins += chunknum * chunklen # storing actual indexes of items of chunk as they are in imeseries
						hiblocks = hibins/detrendlen # again storing index of stds of items which are now in chunk  
                    # Add the candidates (which are sorted by bin)
						
						for bin, val, block, e, ime in zip(hibins, hivals, hiblocks, hiwds, hitds):
							if block not in bad_blocks: # checking for element not to be in bad_region  
								
								
								dm_candlist.append(candidate(ii, val, ime, bin, e)) # storing obects of class candidate 
                                                                                        # in the list dm_ candlist 
                        # Prepare our data for the convolution
							else:
								ind.append(bin)

						if useffts: fftd_chunk = Num.fft.rfft(chunk, axis = -1)  # obtaining discrete fourier transform of chunk in fftd_chunk
                                                                             # only if useffts is true 

    					# Now do the downsampling...
						
						for jj, downfact in enumerate(downfacts): # iterating over list downfacts for getting diff. size of box
                                                               # car functions for convolution
							if useffts: # only if useffts is true
                            # Note:  FFT convolution is faster for _all_ downfacts, even 2
								if (chunknum == numchunks-1):
									goodchunk = fft_convolve(fftd_chunk, fftd_kerns1[jj], overlap, -overlap) # in this way of convolution we first take 
                                                                        # discrete fast fourier transformation of both
                                                                        # boxcar function as well as chunk then we do 
                                                                        # convolution and again we take fourier 
                                                                        # transformation of convolved function that gives
                                                                        # us new goodchunk
								else:
									goodchunk = fft_convolve(fftd_chunk, fftd_kerns[jj], overlap, -overlap)
							else:
                                # The normalization of this kernel keeps the post-smoothing RMS = 1
								kernel = Num.ones(downfact, dtype = Num.float32)
								smoothed_chunk = scipy.signal.convolve(chunk, kernel, 1)
								goodchunk = smoothed_chunk # here we are doing convolution without fourier 
                                                       # transformation
                        #hibins = Num.nonzero(goodchunk>opts.threshold)[0]
							
							
							hibins = Num.flatnonzero(goodchunk>opts.threshold)# Now again we are repeating the same procedure
                                                                            # as we did above upto line (hiblocks = hibins/detrendlen)
							lobins = Num.flatnonzero(goodchunk<=opts.threshold)
							lobins += chunknum * chunklen
							for sk in lobins:
								ind.append(sk) 
							hivals = goodchunk[hibins]
							hiwds = wchunk[hibins]
							hitds = tchunk[hibins]
							hibins += chunknum * chunklen
							hiblocks = hibins/detrendlen
							
							hibins = hibins.tolist()
							hivals = hivals.tolist()
                        # Now walk through the new candidates and remove those
                        # that are not the highest but are within downfact/2
                        # bins of a higher signal pulse

							hibins, hivals, bdlist = prune_related1(hibins, hivals, downfact)
                        # Insert the new candidates into the candlist, but
                        # keep it sorted...
							for bin, val, block, e, ime in zip(hibins, hivals, hiblocks, hiwds, hitds):
								if block not in bad_blocks:
									
									bisect.insort(dm_candlist, candidate(ii, val, ime, bin, e))
 # Now walk through the dm_candlist and remove the ones that
# are within the downsample proximity of a higher
# signal-to-noise pulse	
							
				dm_candlist, bdlist = prune_related2(dm_candlist, downfacts)
				for sk in bdlist:
					ind.append(sk.bin)
				bdm_candlist = []
				for h in ind:
					bdm_candlist.append(candidate(d[h], snr[h], t[h], w[h], 1))
# Write the Bad pulses to an ASCII output file if opts.debug is True
					
				
				if (opts.debug) and len(ind):
					badfile = open(DMstr +'.ascii', mode = 'w')
					for bcand in bdm_candlist:
						badfile.write(str(bcand))
					badfile.close()
					
				
				##########################
				#WHAT TO DO HERE?#########
				##########################

				a = a + len(ind)
# Write the pulses to an ASCII output file
				
				for cand in dm_candlist:
					candlist.append(cand)
				num_v_DMstr[DMstr] = len(dm_candlist)
				
				if (len(dm_candlist)-1):
					for kl in dm_candlist[1:]:
						c.append(kl.sigma)
						DM_.append(kl.DM)
						candlist_.append(kl.sigma)
						ime_.append(kl.ime)
						width_.append(kl.downfact)
						noofpulse_.append(len(dm_candlist))
			else: # If length of List of SNR corresponding to a particular DM is less than 1000
				ind = []
				DMstr = "%.2f"%ii
				N, dt = len(snrz[i]), opts.imesample
				sort_imeseries = sorted(zip(tz[i],dz[i],snrz[i],wz[i]))
				t = []
				d = []
				snr = []
				w = []
				for v in sort_imeseries:
					t.append(v[0])
					d.append(v[1])
					snr.append(v[2])
					w.append(v[3])
				roundN = N
				detrendlen = N # No. of items in a piece to be detrended
				chunklen = N # No. of items in a piece that will undergo association
				overlap = 0
				obsime = N*dt
				if opts.maxwidth > 0.0:
					downfacts = [x for x in default_downfacts if x*dt <= opts.maxwidth]
				else:
					downfacts = [x for x in default_downfacts if x <= max_downfact] # storing reduced form of list 
                                                                                # default_downfact in given list upto 30
				if len(downfacts) == 0:
					downfacts = [default_downfacts[0]]
				if (i == 0):
					orig_N = N
					orig_dt = dt
				if useffts:
					fftd_kerns = make_fftd_kerns(default_downfacts, chunklen) # for the processing of first file


				
				

				numchunks = int(roundN / chunklen) # will be used later in processing when imeseries will be iterated with length
                                                    # N of each chunk
              # Split the imeseries into chunks for detrending
				numblocks = int(roundN / detrendlen) # numblocks  = no. of rows in newly created imeseries below


				imeseries = Num.asarray(snr) # imeseries is list of SNR values for particular DM
				wd = Num.asarray(w) # Values in this list will be later stored as width
				td = Num.asarray(t) # Values in this list will be later stored as width
				
				stds = Num.zeros(numblocks, dtype = Num.float64)
				
				chunk = imeseries  # creating list 'chunk' by copying values in it from imeseries
				
				
				bad_blocks = []
				imeseries = Num.asarray(imeseries)
				bad_blocks = set(bad_blocks)

				dm_candlist = [] # in futher part of programme we will be storing data of class candidate in this list
				for chunknum in range(int(numchunks)): # now we will iterate on imeseries in subparts of size 1000 with chunknum
                                                    # showing index of each subpart
					
					
					chunk = imeseries # creating list 'chunk' by copying values in it from imeseries
					wchunk = Num.asarray(wd)
					tchunk = Num.asarray(td)
                 # Make a set with the current block numbers
					blocks_per_chunk = int(chunklen/detrendlen) # It gives number of stds list in each chunk with size N i.e.
                                                             # we have N/N = 1 std list in each chunk



					lowblock = blocks_per_chunk * chunknum # storing index of first item in list std which is present in chunk
					currentblocks = set(Num.arange(blocks_per_chunk) + lowblock)  # storing index of all items in stds which are 
                                                                             # at presently inside chunk
					localgoodblocks = Num.asarray(list(currentblocks - bad_blocks)) - lowblock# removing all index from currentblocks which 
                                                                           # are also in bad_blocks then subracting lowblocks
                                                                           # from this array to get actual index of good
                                                                           # elements in chunk 
# Search this chunk if it is not all bad

					if len(localgoodblocks):
                        # This is the good part of the data (end effects removed)
						goodchunk = chunk # removing end parts of the data
                    # need to pass blocks/chunklen, localgoodblocks
                    # dm_candlist, dt, opts.threshold to cython routine
                    # Search non-downsampled data first
                    # NOTE:  these nonzero() calls are some of the most
                    #        expensive calls in the program.  Best bet would 
                    #        probably be to simply iterate over the goodchunk
                    #        in C and append to the candlist there.
						
						hibins = Num.flatnonzero(goodchunk>opts.threshold) # storing index in hibins of items in goodchunks  
                                                                        # which are having values greater then threshold
						lobins = Num.flatnonzero(goodchunk<=opts.threshold) # storing index in lobins of items in goodchunks  
                                                                        # which are having values smaller then threshold
						lobins += chunknum * chunklen
						for sk in lobins:
							ind.append(sk)
						hivals = goodchunk[hibins]  # storing values of items in goodchunk with indexes stored in hibins
						hiwds = wchunk[hibins]
						hitds = tchunk[hibins]
						hibins += chunknum * chunklen  # storing actual indexes of items of chunk as they are in imeseries
						hiblocks = hibins/detrendlen # again storing index of stds of items which are now in chunk  
                    # Add the candidates (which are sorted by bin)
						
						
						for bin, val, block, e, ime in zip(hibins, hivals, hiblocks, hiwds, hitds):
							if block not in bad_blocks:  # checking for element not to be in bad_region 
								
								
								dm_candlist.append(candidate(ii, val, ime, bin, e))  # storing obects of class candidate 
                                                                                        # in the list dm_ candlist 
                        # Prepare our data for the convolution
							else:
								ind.append(bin)

						if useffts: fftd_chunk = Num.fft.rfft(chunk, axis = -1)   # obtaining discrete fourier transform of chunk in fftd_chunk
                                                                             # only if useffts is true 

						# Now do the downsampling...
						
						for jj, downfact in enumerate(downfacts):# iterating over list downfacts for getting diff. size of box
                                                               # car functions for convolution
							if useffts: # only if useffts is true
                            # Note:  FFT convolution is faster for _all_ downfacts, even 2
							
								goodchunk = fft_convolve(fftd_chunk, fftd_kerns[jj], overlap, -overlap) # in this way of convolution we first take 
                                                                        # discrete fast fourier transformation of both
                                                                        # boxcar function as well as chunk then we do 
                                                                        # convolution and again we take fourier 
                                                                        # transformation of convolved function that gives
                                                                        # us new goodchunk
							else:
                                 # The normalization of this kernel keeps the post-smoothing RMS = 1
								kernel = Num.ones(downfact, dtype = Num.float32)
								smoothed_chunk = scipy.signal.convolve(chunk, kernel, 1)
								goodchunk = smoothed_chunk # here we are doing convolution without fourier 
                                                       # transformation
                        #hibins = Num.nonzero(goodchunk>opts.threshold)[0]
							
							hibins = Num.flatnonzero(goodchunk>opts.threshold)# Now again we are repeating the same procedure
                                                                            # as we did above upto line (hiblocks = hibins/detrendlen)
							
							lobins = Num.flatnonzero(goodchunk<=opts.threshold)
							lobins += chunknum * chunklen
							for sk in lobins:
								ind.append(sk)
							hivals = goodchunk[hibins]
							hiwds = wchunk[hibins]
							hitds = tchunk[hibins]
							hibins += chunknum * chunklen
							hiblocks = hibins/detrendlen
							
							hibins = hibins.tolist()
							hivals = hivals.tolist()
                        # Now walk through the new candidates and remove those
                        # that are not the highest but are within downfact/2
                        # bins of a higher signal pulse

							hibins, hivals, bdlist = prune_related1(hibins, hivals, downfact)
                        # Insert the new candidates into the candlist, but
                        # keep it sorted...
							for sk in bdlist:
								ind.append(sk)

							for bin, val, block, e, ime in zip(hibins, hivals, hiblocks, hiwds, hitds):
								if block not in bad_blocks:
									
									bisect.insort(dm_candlist, candidate(ii, val, ime, bin, e))
# Now walk through the dm_candlist and remove the ones that
# are within the downsample proximity of a higher
# signal-to-noise pulse		
							
				dm_candlist, bdlist = prune_related2(dm_candlist, downfacts)
				for sk in bdlist:
					ind.append(sk.bin)
				bdm_candlist = []
				for h in ind:
					bdm_candlist.append(candidate(d[h], snr[h], t[h], w[h], 1))
					
# Write the Bad pulses to an ASCII output file if opts.debug is True					
					
				
				if (opts.debug) and len(ind):
					badfile = open(DMstr +'.ascii', mode = 'w')
					for bcand in bdm_candlist:
						badfile.write(str(bcand))
					badfile.close()

				a = a + len(ind)
# Write the pulses to an ASCII output file
				
				
				for cand in dm_candlist[1:]:
					candlist.append(cand)
				num_v_DMstr[DMstr] = len(dm_candlist)
				if (len(dm_candlist)-1):
					for kl in dm_candlist[1:]:
						c.append(kl.sigma)
						DM_.append(kl.DM)
						candlist_.append(kl.sigma)
						ime_.append(kl.ime)
						width_.append(kl.downfact)
						noofpulse_.append(len(dm_candlist))
		p = float(a)/float(len(DM)) *100
 
		c.sort() 			##Corerection20190415 One tab is extra given 
		smedian = c[int(len(c)/2)]	##
						##
		if sm:				##
			ls = smedian		##

		
	DM__ = []
	candlist__ = []
	ime__ = []
	width__ = []
	noofpulse__ = []
	

	
	for jl, lk in enumerate(DM_):
		if (ld <= DM_[jl] <= ud and ls <= candlist_[jl] <= us and lt <= ime_[jl] <= ut and lw <= width_[jl] <= uw):
			DM__.append(DM_[jl])
			candlist__.append(candlist_[jl])
			ime__.append(ime_[jl])
			width__.append(width_[jl])
			noofpulse__.append(noofpulse_[jl])
	DM__ = Num.asarray(DM__)
	candlist__ = Num.asarray(candlist__)
	ime__ = Num.asarray(ime__)
	width__ = Num.asarray(width__)
	noofpulse__ = Num.asarray(noofpulse__)

	CC = candlist__.copy() # (addition 0) 
	CC[CC>cs] = cs # (addition 0)

	
	plo = plt.figure()
	ax1 = plo.add_subplot(321)
	ax2 = plo.add_subplot(322)
	ax3 = plo.add_subplot(325)
	ax4 = plo.add_subplot(326)

	aa = ax4.scatter(ime__ ,DM__ ,c = CC, s = (candlist__**ef)/5000)
	ax4.set_title("DM vs ime Scatter (SNR in color)")
	ax4.set_xlabel("ime")
	ax4.set_ylabel("DM")
	ax4.axis([min(ime__)-0.5, max(ime__)+0.5, min(DM__)-5, max(DM__)+5])	
	bb = ax1.scatter(DM__, width__, c = CC, s = (candlist__**ef)/5000)
	ax1.set_title("Width vs DM Scatter (No. of Pulses in color)")
	ax1.set_ylabel("Width (ms)")
	ax1.set_xlabel('DM')
	ax1.axis([min(DM__)-5,max(DM__)+5,min(width__)-0.2,max(width__)+0.5])
	cc = ax2.scatter(DM__, width__, c = noofpulse__, s = noofpulse__)
	ax2.set_xlabel("DM")
	ax2.set_ylabel("Width (sample)")
	ax2.set_title("Width vs DM Scatter")
	ax2.axis([min(DM__)-5,max(DM__)+5,min(width__)-0.2,max(width__)+0.5])
	ax3.plot(DM__, candlist__, "o", markersize = 0.5)
	ax3.set_ylabel("SNR")
	ax3.set_xlabel("DM")
	ax3.set_title("SNR vs DM Scatter")
	plo.colorbar(bb)
	plo.colorbar(cc)
	end = time.time()
	print "Run Time of code in seconds is " + str(end - start)
	plt.show()


	

else:

	lt = opts.lime
	ut = opts.uime
	ld = opts.ldm
	ud = opts.udm
	ls = opts.lsnr
	us = opts.usnr
	lw = opts.lwidth
	uw = opts.uwidth
	ef = opts.efactor # (addition 0) 
	cs = opts.csaturation # (addition 0)

	f = Num.fromfile(args[0], dtype = Num.float32)
	l = len(f)
	DM =[]
	ime =[]
	SNR = []
	width = []
	for i in range(0,l,4):
		DM.append(f[i])
		ime.append(f[i+1])
		SNR.append(f[i+2])
		width.append(f[i+3])
	DM_sorted = []
	for i in DM:
		if i not in DM_sorted:
			DM_sorted.append(i)

	DM_sorted.sort()
	d = []
	for i in range(len(DM_sorted)):
		d.append([])


	D = []
	T = []
	C = []
	W = []
	N = []
	S = []

	D_ = []
	T_ = []
	C_ = []
	W_ = []
	N_ = []
	S_ = []

	for hh, h in enumerate(DM_sorted):
		for jl, lk in enumerate(DM):
			if (ld <= DM[jl] <= ud and ls <= SNR[jl] <= us and lt <= ime[jl] <= ut and lw <= width[jl] <= uw and DM_sorted[hh] == DM[jl]):
				d[hh].append(DM[jl])
				print DM[jl]
	F = []
	for uu in d:
		F.append(uu[0])
		

	for jl, lk in enumerate(DM):
		if (ld <= DM[jl] <= ud and ls <= SNR[jl] <= us and lt <= ime[jl] <= ut and lw <= width[jl] <= uw):
			D_.append(DM[jl])
			C_.append(SNR[jl])
			T_.append(ime[jl])
			W_.append(width[jl])
			S_.append(4)
			N_.append(len(d[F.index(DM[jl])]))
			
			
			
	
	l = sorted(zip(C_, D_, T_, W_, N_), reverse = True)

	for i,ii in enumerate(l):
				D.append(ii[1])
				C.append(ii[0])
				W.append(ii[3])
				T.append(ii[2])
				N.append(ii[4])
				S.append(4)

	D = Num.asarray(D)
	C = Num.asarray(C)
	T = Num.asarray(T)
	W = Num.asarray(W)
	N = Num.asarray(N)

	CC = C.copy() # (addition 0) 
	CC[CC>cs] = cs # (addition 0)


	plo = plt.figure()
	ax1 = plo.add_subplot(321)
	ax2 = plo.add_subplot(322)
	ax3 = plo.add_subplot(325)
	ax4 = plo.add_subplot(326)

	aa = ax4.scatter(T ,D ,c = CC, s = (C**ef)/5000) # (addition 0)
	ax4.set_title("DM vs ime Scatter (SNR in color)") # (addition 0)
	ax4.set_xlabel("ime")
	ax4.set_ylabel("DM")
	ax4.axis([min(T)-0.5, max(T)+0.5, min(D)-5, max(D)+5])	
	bb = ax1.scatter(D, W, c = CC, s = (C**ef)/5000) # (addition 0)
	ax1.set_title("Width vs DM Scatter (No. of Pulses in color)") # (addition 0)
	ax1.set_ylabel("Width (sample)")
	ax1.set_xlabel('DM')
	ax1.axis([min(D)-5,max(D)+5,min(W)-0.2,max(W)+0.5])
	cc = ax2.scatter(D, W, c = N, s = N)
	ax2.set_xlabel("DM")
	ax2.set_ylabel("Width (sample)")
	ax2.set_title("Width vs DM Scatter (Np in color)") # (addition 0)
	ax2.axis([min(D)-5,max(D)+5,min(W)-0.2,max(W)+0.5])
	ax3.plot(D, C, "o", markersize = 0.5)
	ax3.set_ylabel("SNR")
	ax3.set_xlabel("DM")
	ax3.set_title("SNR vs DM Scatter")
	plo.colorbar(bb)
	plo.colorbar(cc)
	end = time.time()
	print "Run Time of code in seconds is " + str(end - start)
	plt.show()


