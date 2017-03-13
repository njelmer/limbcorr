#!/usr/local/bin/python

'''
nje_limbcorr_v2

Author:	Nicholas J. Elmer
	Department of Atmospheric Science, University of Alabama in Huntsville (UAH)
	NASA Short-term Prediction Research and Transition (SPoRT) Center
	Huntsville, Alabama, USA
	nicholas.j.elmer@nasa.gov

Version: 2.0  (Last updated June 2016)
	Note: The function call is different than in Version 1.x

Revision history:
	Mar 2015	Version 1.0 released
	Jan 2016	Enabled limb correction of projected data (v1.1)
	May 2016	Himawari AHI correction coefficients added
			Created tables for additional infrared bands, 
				not just the common RGB bands
			Modified function arguments to include satellite name to
				support more instruments
	Jun 2016	Version 2.0 released

Description:
	Limb correct polar and geostationary infrared imagery

Input:
 Required:
   data:	2D or 3D array,	LxPxB array of brightness temperatures
					L = number of scans, 
					P = number of pixels per scan, 
					B = number of bands
					The data can be in any projection.

   lats: 	array (2D),	latitude array of full swath/image

   vza: 	array (2D),	array of viewing zenith angles for full swath/image (radians)

   satellite:	string,		name of satellite
					Valid options for each sensor:
						SENSOR		SATELLITE
						=========	=========================
						abi		goes-r
						ahi		himawari
						avhrr		noaa-15, noaa-18, noaa-19,
									 metop-a, metop-b
						modis		aqua, terra
						seviri		meteosat
						viirs*		snpp
					The satellite name is only used for intercalibration.
					*Moderate Resolution bands only (M12-M16)
						

   sensor: 	string,		name of sensor/instrument
					Valid options: abi, ahi, avhrr, modis, seviri, viirs

   jday: 	integer/float,	Julian day / day of year

   bandorder: 	int array,	list of bands in same order as 3D data array (Ex. [12,14,15]).
					If an unsupported band is listed, that band is not 
					corrected and is returned unchanged.
 Optional:
   fillval: 	float,		Fill value for masking bad values (e.g., points beyond full disk, 
					negative brightness temperatures, etc.).  Default is nan.

   highres: 	boolean,	Performs a high-spatial-resolution limb correction,
					i.e., the full latitude array is used. By default,
					only the latitude of the center pixel of each scan
					is used to speed up processing, thereby assuming that
					the latitude of an entire scan is constant.
					Note that the highres option takes approximately
					100-200 times longer than the default.

   ctp: 	array (2D),	Cloud top pressure in hPa, which must be the same resolution
					as the data array.  If cloud top pressure is not provided,
					all pixels are assumed to be cloudfree, i.e., the full 
					optical pathlength is corrected for limb effects.

   tblpath: 	string,		path containing the limb correction tables.
					This path will match the path containing the README file 
					when the release is first downloaded.
					Default is './nje_limbcorr_v2.0/tables/'.

   refsen: 	string,      	reference sensor ('ahi', 'abi', 'seviri')
					Used for intercalibration to reference sensor 
					using methodology in Elmer et al. (2016; see README
					for full citation).  By default, the brightness
					temperatures are intercalibrated to SEVIRI.
					To prevent intercalibration, use the null string ''.

Returns:
  An array of limb corrected brightness temperatures of the same size as the input data array.

Required Python modules:
	itertools
	numpy
	time
	os

Internal Functions:
	calc_coeff
	calc_cld


Notes on intecalibration:
	- Intercalibration values are only available for IR bands listed in Elmer et al. (2016).
		For all other channels, no intercalibration is performed.
	- GOES-R ABI is not yet operational, so intercalibration values w.r.t. other sensors
		cannot be calculated empirically.  Since ABI is very similar to AHI, 
		AHI intercalibration values are used when ABI is specified.
	- Intercalibration values may change over time as sensors age or instrument calibration
		is updated.

Example::
from nje_limbcorr_v2 import limbcorr
import numpy as np

band27 = np.array([ [270., 270., 270.],[270., 270., 270.],[270., 270., 270.] ])
band28 = np.array([ [280., 280., 280.],[280., 280., 280.],[280., 280., 280.] ])
band31 = np.array([ [290., 290., 290.],[290., 290., 290.],[290., 290., 290.] ])
band32 = np.array([ [300., 300., 300.],[300., 300., 300.],[300., 300., 300.] ])

data      = np.dstack(( band28, band27, band32, band31))
latitudes = np.array([ [35., 35., 35.],[34., 34., 34.],[33., 33., 33.] ])
vza       = np.array([ [80.,  0., 80.],[80.,  0., 80.],[80.,  0., 80.] ])
cldtp     = np.array([ [100.,  600., 1000.],[600.,  650., 700.],[300.,  400., 500.] ])

# Without cloud correction
no_cldcorr = limbcorr(data, latitudes, vza*(np.pi/180.), 'meteosat', 'seviri', 320, [5, 6, 32, 31])
# With cloud correction
yes_cldcorr = limbcorr(data, latitudes, vza, 'aqua', 'modis', 320, [28, 27, 32, 31], \
			refsen='ahi', highres=True, ctp=cldtp, \
			tblpath='~/mypath/nje_limbcorr_v2.0/tables/', fillval=0.0)
'''


import numpy as np
import time
import itertools
import os


# ==== INTERNAL FUNCTIONS ========================================
# Calculate limb correction coefficients
def calc_coeff(ma, mb, jday, lats, order=9):
    coeffA = np.zeros_like(lats)
    coeffB = np.zeros_like(lats)
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
         value = jday**i * lats**j
         coeffA += ma[k] * value
         coeffB += mb[k] * value
    return (coeffA, coeffB)

# Calculate cloud correction coefficients
def calc_cld(mc, jday, lats, order=9):
    jday = jday/100.
    lats = lats/100.
    cld = np.zeros_like(lats)
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
         value = jday**i * lats**j
         cld += mc[k] * value
    return cld/100.
# ==============================================================

# Main Limb Correction Function
def limbcorr(data, lats, vza, satellite, sensor, jday, bandorder, \
		refsen='seviri', highres=False, ctp=[], \
		tblpath='./nje_limbcorr_v2.0/tables/', fillval=np.nan):

  # Print input information
  print '\nSat. & Sensor  : %s %s' %(satellite, sensor)
  print 'Day of year    : %03i' %(jday)
  print 'Bands          : %s' %(bandorder)
  print 'Ref. Sensor    : %s' %(refsen)
  print 'Data Shape     :', (np.shape(data))
  print 'Lat/Lon Shape  :', (np.shape(lats))
  print 'VZA Shape      :', (np.shape( vza))

  # Set up cloud correction, if needed
  if ctp == []: cldcorr = False
  else: 
   cldcorr = True
   pressures = np.array([0.10,    0.29,    0.51,    0.80,    1.15, \
           1.58,    2.08,    2.67,    3.36,    4.19,    5.20, \
           6.44,    7.98,    9.89,   12.26,   15.19,   18.82, \
          23.31,   28.88,   35.78,   44.33,   54.62,   66.62, \
          80.40,   95.98,  113.42,  132.76,  153.99,  177.12, \
         202.09,  228.84,  257.36,  287.64,  319.63,  353.23, \
         388.27,  424.57,  461.90,  500.00,  538.59,  577.38, \
         616.04,  654.27,  691.75,  728.16,  763.21,  796.59, \
         828.05,  857.34,  884.27,  908.65,  930.37,  949.35, \
         965.57,  979.06,  989.94,  998.39, 1004.64, 1009.06])
   diff  = (pressures[1:] - pressures[0:-1])/2.
   upper = np.append(pressures[0:-1] + diff, [1013.0])
   lower = np.append([0.0], pressures[1:] - diff)
   # Normalize cld top pressures to pressures defined above
   ctp = ctp * (np.nanmax(ctp)/pressures[-1])

  # Check that tblpath is valid
  if not os.path.exists(tblpath): raise ValueError('Path to tables does not exist!')

  # Check that satellite name is valid
  valid = ['goes-r', 'himawari', 'noaa-15', 'noaa-18', 'noaa-19', \
	   'metop-a', 'metop-b', 'aqua', 'terra', 'meteosat', 'snpp']
  check = satellite in valid
  if not check: raise ValueError('Invalid satellite name')

  # Check that satellite/sensor combination is valid and define parameters
  earthrad = 6371.			    #km; radius of earth
  if   sensor == 'abi'   : 
       satalt = 35800.  #km
       if satellite != 'goes-r':
          raise ValueError('Invalid satellite/sensor combination')
  elif sensor == 'ahi'   : 
       satalt = 35800.  #km
       if satellite != 'himawari':
          raise ValueError('Invalid satellite/sensor combination')
  elif sensor == 'avhrr' : 
       satalt =   833.  #km
       if satellite != 'noaa-15' and satellite != 'noaa-18' and \
          satellite != 'noaa-19' and \
          satellite != 'metop-a' and satellite != 'metop-b':
          raise ValueError('Invalid satellite/sensor combination')
  elif sensor == 'modis' : 
       satalt =   705.  #km; altitude
       if ((satellite != 'aqua') and (satellite != 'terra')):
          raise ValueError('Invalid satellite/sensor combination')
  elif sensor == 'seviri': 
       satalt = 35800.  #km
       if satellite != 'meteosat':
          raise ValueError('Invalid satellite/sensor combination')
  elif sensor == 'viirs':
       satalt =   824.  #km
       if satellite != 'snpp':
          raise ValueError('Invalid satellite/sensor combination')
  else: raise ValueError('Invalid sensor name')


  # Check that data is 2D or 3D.  Expand dimensions to 3D if input is 2D.
  sz = np.shape(data)
  ndims = len(sz)
  if ndims != 2 and ndims != 3: 
	raise Exception('Data array must be 2D (for single band) or 3D (for multiple bands)')
  if ndims == 2: data = np.expand_dims(data, 2)
  sz = np.shape(data)
  lines = sz[0]
  pixels = sz[1]
  nbands = len(bandorder)
  if len(bandorder) != sz[2]: 
    raise Exception('Bandorder does not match the number of bands in data array')

  # Make sure latitude array is 2D and same size as imagery
  szl = np.shape(lats)
  lndims = len(szl)
  if szl[0] != sz[0] or szl[1] != sz[1] or lndims != 2: 
    raise IndexError('Latitude array must be 2D and the same resolution as the imagery')

  # Make sure vza is 2D and the same size as latitude array
  szvza = np.shape(vza)
  andims = len(szvza)
  if szvza != szl or szvza[0] != sz[0] or szvza[1] != sz[1] or lndims != 2: 
    raise IndexError('VZA array must be 2D and the same resolution as the imagery')
  # Convert to radians if given in degrees
  if np.nanmax(vza) > 2.0: 
    vza = vza*3.141592654/180.
    print '   VZA provided in degrees. Converting to radians...'

  # Define array for limb corrected brightness temperatures
  correct = np.zeros_like(data)

  # Prepare Julian day and latitudes
  # NOTE: jday and lats were divided by 100 when coefficients were calculated,
  #       so the same is done here in order to retrieve the correct values.
  ddd = float(jday)/100.
  ttt = np.copy(lats)
  badval = np.where( (np.abs(lats) > 90.) | (np.isnan(lats) == 1) )
  ttt[badval] = np.nan
  lats = None
  # Calculate mean latitude per image line (to speed up processing)
  if not highres:
    ttt = np.nanmean(ttt, axis=1)
  # Clip latitudes to [-72.5, 72.5], since coefficients are not defined beyond this range.
  ttt[ttt >=  72.5] =  72.5
  ttt[ttt <= -72.5] = -72.5
  ttt = ttt/100.

  # Limit VZA from 0 to 89 deg (0 to pi/2 rad)
  vza[badval] = np.nan
  vza[vza > 89.*np.pi/180.] = 89.*np.pi/180.

  # Calculate vza-based factor array
  factor = np.abs(np.log(np.cos(vza)))
  print '   Min/Max VZA : %5.3f  %5.3f  radians\n' %(np.nanmin(vza), np.nanmax(vza))
  factor[badval] = np.nan
  vza = None

  # Begin limb correction
  print 'Performing limb correction...'
  if highres: print 'High resolution correction: ON'
  else:       print 'High resolution correction: OFF'
  for m in range(nbands):
    print '...Band %02i...' %(bandorder[m])

    # Make sure band is a supported thermal infrared band. 
    #	If not, copy data without limb-correcting.
    if sensor == 'abi'    and (bandorder[m] < 7 or bandorder[m] >16): 
       correct[:,:,m] = data[:,:,m]
       continue
    if sensor == 'ahi'    and (bandorder[m] < 7 or bandorder[m] >16): 
       correct[:,:,m] = data[:,:,m]
       continue
    if sensor == 'avhrr'  and (bandorder[m] < 3 or bandorder[m] > 5):  
       correct[:,:,m] = data[:,:,m]
       continue
    if sensor == 'modis'  and (bandorder[m] < 27 or bandorder[m] > 36) \
			  and (bandorder[m] != 20) : 
       correct[:,:,m] = data[:,:,m]
       continue
    if sensor == 'seviri' and (bandorder[m] < 4 or bandorder[m] >11): 
       correct[:,:,m] = data[:,:,m]
       continue
    if sensor == 'viirs'  and (bandorder[m] < 12 or bandorder[m] > 16): 
       correct[:,:,m] = data[:,:,m]
       continue

    # Read lookup tables
    # Version 2.0
    try:
      coeffAfile = os.path.join(tblpath,sensor,'%s_band%02i_Acoeff.txt' \
		%(sensor,bandorder[m]))
      coeffBfile = os.path.join(tblpath,sensor,'%s_band%02i_Bcoeff.txt' \
		%(sensor,bandorder[m]))
      ma = np.genfromtxt(coeffAfile, dtype=None, skip_header=1, autostrip=True)
      mb = np.genfromtxt(coeffBfile, dtype=None, skip_header=1, autostrip=True)
      if cldcorr:
        cldfile  = os.path.join(tblpath,sensor,'%s_band%02i_CLDscl.txt' \
		%(sensor,bandorder[m]))
        mc = np.genfromtxt(cldfile, dtype=None, skip_header=1, autostrip=True)
    # For Version 1.0 compatibility
    except IOError:
      coeffAfile = os.path.join(tblpath,'%s_band%02i_Acoeff.txt' \
		%(sensor,bandorder[m]))
      coeffBfile = os.path.join(tblpath,'%s_band%02i_Bcoeff.txt' \
		%(sensor,bandorder[m]))
      ma = np.genfromtxt(coeffAfile, dtype=None, skip_header=1, autostrip=True)
      mb = np.genfromtxt(coeffBfile, dtype=None, skip_header=1, autostrip=True)
      if cldcorr:
        cldfile  = os.path.join(tblpath,'%s_band%02i_CLDscl.txt' \
		%(sensor,bandorder[m]))
        mc = np.genfromtxt(cldfile, dtype=None, skip_header=1, autostrip=True)

    # Calculate correction coefficients
    # Calculate C2 (tb_qcoeff) and C1 (tb_lcoeff)
    start = time.clock()
    tb_qcoeff, tb_lcoeff = calc_coeff(ma, mb, ddd, ttt)
    print '   C2 range: [%5.3f, %5.3f]' %(np.nanmin(tb_qcoeff), np.nanmax(tb_qcoeff))
    print '   C1 range: [%5.3f, %5.3f]' %(np.nanmin(tb_lcoeff), np.nanmax(tb_lcoeff))
    # Calculate Q (cldfactor)
    cldfactor = np.ones([lines,pixels],dtype=float)
    if cldcorr:
       for plev in range(np.shape(mc)[0]):
         cf = calc_cld(mc[plev,:], ddd, ttt)
         # Fill cldfactor array with appropriate Q values
         for line in range(lines):
           ind = np.where((ctp[line,:] >= lower[plev]) & (ctp[line,:] < upper[plev]))
           if len(ind[0]) == 0: continue
           try: cldfactor[line,ind] = cf[line]
           except ValueError: cldfactor[line,ind] = cf[line,ind]

       over = np.where(cldfactor > 1.0)
       cldfactor[over] = 1.0
    end = time.clock()
    print '   Calculation time: %5.1f seconds\n' %(end-start)

    # Get offset value for intercalibration:
    tb_offset = 0.00	# tb_offset = 0.0 if value not found below
    # Intercalibration to SEVIRI
    if refsen == 'seviri':
      # modis
      if sensor == 'modis':
        if satellite == 'aqua':
          if bandorder[m] == 20:  tb_offset = -2.25
          if bandorder[m] == 27:  tb_offset = -3.10
          if bandorder[m] == 28:  tb_offset =  0.10
          if bandorder[m] == 29:  tb_offset = -0.10
          if bandorder[m] == 30:  tb_offset = -1.60
          if bandorder[m] == 31:  tb_offset =  0.15
          if bandorder[m] == 32:  tb_offset = -0.30
        if satellite == 'terra':
          if bandorder[m] == 20:  tb_offset = -2.25
          if bandorder[m] == 27:  tb_offset =  0.00
          if bandorder[m] == 28:  tb_offset =  0.48
          if bandorder[m] == 29:  tb_offset = -0.90
          if bandorder[m] == 30:  tb_offset = -1.55
          if bandorder[m] == 31:  tb_offset =  0.25
          if bandorder[m] == 32:  tb_offset = -0.30
      # viirs-m
      if sensor == 'viirs':
        if satellite == 'snpp':
          if bandorder[m] == 12:  tb_offset = -3.00
          if bandorder[m] == 14:  tb_offset =  0.70
          if bandorder[m] == 15:  tb_offset =  0.00
          if bandorder[m] == 16:  tb_offset = -0.40
      # avhrr
      if sensor == 'avhrr':
        if satellite == 'noaa-15' or satellite == 'noaa-18':
          if bandorder[m] ==  3:  tb_offset = -3.00
          if bandorder[m] ==  4:  tb_offset = -0.40
          if bandorder[m] ==  5:  tb_offset = -0.20
        if satellite == 'noaa-19':
          if bandorder[m] ==  3:  tb_offset = -3.00
          if bandorder[m] ==  4:  tb_offset = -0.50
          if bandorder[m] ==  5:  tb_offset = -0.20
        if satellite == 'metop-a':
          if bandorder[m] ==  3:  tb_offset = -2.50
          if bandorder[m] ==  4:  tb_offset = -0.20
          if bandorder[m] ==  5:  tb_offset = -0.20
        if satellite == 'metop-b':
          if bandorder[m] ==  3:  tb_offset = -2.75
          if bandorder[m] ==  4:  tb_offset = -0.30
          if bandorder[m] ==  5:  tb_offset = -0.20

    # Intercalibration to AHI
    elif refsen == 'ahi' or refsen == 'abi':
      # modis
      if sensor == 'modis':
        if satellite == 'aqua':
          if bandorder[m] == 20:  tb_offset =  0.00
          if bandorder[m] == 27:  tb_offset = -3.80
          if bandorder[m] == 28:  tb_offset =  0.20
          if bandorder[m] == 29:  tb_offset =  0.00
          if bandorder[m] == 30:  tb_offset = -3.40
          if bandorder[m] == 31:  tb_offset =  0.70
          if bandorder[m] == 32:  tb_offset =  0.00
        if satellite == 'terra':
          if bandorder[m] == 20:  tb_offset =  0.00
          if bandorder[m] == 27:  tb_offset = -3.80
          if bandorder[m] == 28:  tb_offset =  0.20
          if bandorder[m] == 29:  tb_offset =  0.00
          if bandorder[m] == 30:  tb_offset = -3.40
          if bandorder[m] == 31:  tb_offset =  0.70
          if bandorder[m] == 32:  tb_offset =  0.00
      # viirs-m - not yet calculated for AHI
      if sensor == 'viirs':
        if satellite == 'snpp':
          if bandorder[m] == 12:  tb_offset =  0.00
          if bandorder[m] == 14:  tb_offset =  0.00
          if bandorder[m] == 15:  tb_offset =  0.00
          if bandorder[m] == 16:  tb_offset =  0.00
      # avhrr
      if sensor == 'avhrr':
        if satellite == 'noaa-15' or satellite == 'noaa-18':
          if bandorder[m] ==  3:  tb_offset = -0.20
          if bandorder[m] ==  4:  tb_offset =  0.00
          if bandorder[m] ==  5:  tb_offset = -2.20
        if satellite == 'noaa-19':
          if bandorder[m] ==  3:  tb_offset = -0.20
          if bandorder[m] ==  4:  tb_offset = -0.20
          if bandorder[m] ==  5:  tb_offset = -2.20
        if satellite == 'metop-a':
          if bandorder[m] ==  3:  tb_offset = -0.20
          if bandorder[m] ==  4:  tb_offset = -0.20
          if bandorder[m] ==  5:  tb_offset = -2.75
        if satellite == 'metop-b':
          if bandorder[m] ==  3:  tb_offset = -0.20
          if bandorder[m] ==  4:  tb_offset = -0.20
          if bandorder[m] ==  5:  tb_offset = -2.50


    # Expand dimensions of correction coefficient arrays to match data if
    #	highres correction was not done
    if not highres:
      tb_qcoeff = np.expand_dims(tb_qcoeff, 2)
      tb_qcoeff = np.tile(tb_qcoeff, (1,sz[1]))
      tb_lcoeff = np.expand_dims(tb_lcoeff, 2)
      tb_lcoeff = np.tile(tb_lcoeff, (1,sz[1]))

    # Calculate corrected brightness temperatures using Eqn. 6 from Elmer et al. (2016)
    corr = (np.squeeze(data[:,:,m]) + tb_offset) + \
		cldfactor*(tb_lcoeff*factor + tb_qcoeff*(factor**2))

    corr[badval]=fillval
    correct[:,:,m] = corr

  return np.squeeze(correct)
  print 'Done!'








