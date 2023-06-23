import numpy as np
import pydicom
import os

input_dcm_dir  = '../data/ncRecon'
#file_name = "1_BWHBI87_LH_Str_1st_[ncRecon]"
filenames = [os.path.join(input_dcm_dir, fn) for fn in os.listdir(input_dcm_dir) if fn.endswith("IMA")]
for filename in filenames:
	ds = pydicom.dcmread(filename)
	vol = ds.pixel_array # shape = (slice, column, row)
	with open(filename[:-4] + ".vol", "wb") as f:
		for slice in vol:
			slice.astype('float32').tofile(f)
