## The ideas from "check_probability" shows that all the thresholds
## (confidences) should be set at roughly 0.3. However, those 
## analysis are based on several unclear assumptions.
## For example, it assumes that the F-measure or F-measure ratios
## can be treated as the weight sum of all possible F-measures
## for all possible initial predictions, which cannot be proved
## true in general. 

## However, the good news is, it seems that the F-measure is really 
## really in a reasonable region of chosen threshold of new tags and
## related tags.

number_popular_tag = 6000

threshold_newtag = 0.28
threshold_relatedtag = 0.33 * 100

## first 10000 elements in Train.csv
## (horizontal newtag; vertical relatedtag):
## all tag information from the "title" of the questions
##			0.20			0.25			0.27			0.28			0.29			0.30			0.32			0.35
##	0.20													0.336656075036
##	0.25													0.338643542014
##	0.30	0.338380511711	0.339053866689	0.340098772894	0.339924090354	0.339982596848	0.340177820513	0.338911983572	339975829171
##	0.32													0.339413044178
##	0.35													0.339675143745
##	0.40													0.339970100455
##	0.50													0.335001002331
##	0.60													0.330565501721

##------------------------------------------------------------------

## The minimal number of "countYes" for a tag if it is to be considered
number_body_tag = 500
threshold_countYes_body = 50
