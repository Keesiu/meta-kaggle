import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from load_toys_rev import *

filename = 'data/toys_rev2_sample10.csv'
sample_factor = 10
toy_list = load_toys_rev(filename)

overall_minutes = 0
count_day_per_minute = [0]*(24*60)
count_year_per_hour = [0]*(365*24)
hours_need_per_hour = [0]*(365*24)

for toy in toy_list:
    overall_minutes += toy.duration*sample_factor
    count_day_per_minute[toy.minutes_from_day-1] += 1*sample_factor
    count_year_per_hour[int((toy.minutes_from_year-1)/60)] += 1*sample_factor
    hours_need_per_hour[int((toy.minutes_from_year-1)/60)] += toy.duration/60*sample_factor

print("overall minutes = ", overall_minutes) #26010942390
print("%*(10/24)/(365*24*60) = ", overall_minutes*10/24/(365*24*60)) #20620.039312214612

"""
In[5]:= Solve[1270712461.7873 == tf*365*24*60*Log[1 + 900], tf]
Out[5]= {{tf -> 355.352}}
In[3]:= 20620.039312214612/900
Out[3]= 22.9112
"""

plt.subplot(311)
plt.plot(count_day_per_minute, "r.")
plt.xlabel('minute of the day')
plt.ylabel('count')
#plt.title('distribution of the gifts compare to minute of the day')

plt.subplot(312)
plt.plot(count_year_per_hour, "b.")
plt.xlabel('hour of the year')
plt.ylabel('count')
#plt.title('distribution of the gifts in a year')

plt.subplot(313)
plt.semilogy(hours_need_per_hour, "g.")
plt.xlabel('hour of the year')
plt.ylabel('hours of work produced per hour')
#plt.title('distribution of the gifts in a year')

plt.show()
