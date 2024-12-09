#mode
from collections import Counter
n_num=[1,2,3,4,5,5]
n=len(n_num)
data=Counter(n_num)
get_mode=dict(data)
mode=[k for k,v in get_mode.items() if v==max(list(data.values()))]
if len(mode)==n:
    get_mode="no mode found"
else:
    get_mode="mode is/ are:"+'.'.join(map(str,mode))
print(get_mode)
