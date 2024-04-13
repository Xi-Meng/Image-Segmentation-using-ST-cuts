## Project Description:

Advanced algorithm design and analysis course assignment
The idea of maximum flow and minimum cut set is used to segment the image

## Dependencies:
require python 3.8

All dependencies are enlisted in requirements.txt
	
	pip install -r requirements.txt	

**Note:**

Extension need not be png
	
**Runtime Commands**

**fast_seg.py**
The main code used to produce the segmentation results. command-line arguments
1. **-i / --img** : -i <path to input image>
2. **-a / --algo** : values “bk”/”ff”
3. **“bk”** - used to perform segmentation using boykov kolmogorov algorithm
4. **“ff”** - used to perform segmentation using ford fulkerson algorithm -s/ --sp_en : values“y”/”n”

**Example:** python fast_seg.py -i ./images/bunny.png -a bk


## Results 

![alt text](output/result1.png)

![alt text](output/result2.png)