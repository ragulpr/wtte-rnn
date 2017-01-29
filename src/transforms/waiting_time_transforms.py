import numpy as np
"""
Functions  to calculate time-since-event created from an 
array of events (use steps_since_true) or indexes indicating events and a window
Should produce something like:
    i            |0|1|2|3|4|5|
    event        |0|0|1|1|0|1|
    steps_since: |1|2|0|0|1|0| (count_up_to = False)
    steps_since: |1|2|3|1|1|2| (count_up_to = True)
    censoring :  |1|1|0|0|0|0| (count_up_to = False)
    censoring :  |1|1|1|0|0|0| (count_up_to = True)
And if a window is chosen it just picks that window without ever
creating the whole array.

OBS : not tested yet, not working yet and needs to be rebuilt from scratch. 
"""

def steps_since_true(event,count_up_to = False, 
                     return_censoring = False, 
                     init_steps_since_true = 1):
    """time-since-event window created from an array indicating events

    Args:
        event      : 1d int array of indexes pointing out events
        count_up_to: If True the count is reset to 1 after step of event
        return_censoring : If true returns tuple of (steps_since_true,censoring)
        init_steps_since_true: Initial count. If 1 we assume no prior events
          if  >1 assume events occuring init_steps_since_true-steps prior
        
        Note that init_steps_since_true=1 always induces right-censoring 
        up to first event when count_up_to =True

    Returns (if return_censoring tuple of both):
        steps_since: np.int32 array
        is_censored: np.8     array indicating censoring

    (TODO)Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    x = np.int32(event)
    z = -init_steps_since_true

    for i in xrange(len(event)):
        if event[i]>0:
            z = i
        x[i] = i-z
        
    if return_censoring:
        if count_up_to :
            # If no event happened prior: censor up to first observed event
            is_censored    = np.roll(event,1)
            is_censored[0] = 0 if init_steps_since_true==1 else 0
        else: 
            # If no event happened prior: censor until first observed event
            is_censored = np.int8(event)
            if init_steps_since_true>1:
                is_censored[0] = 1
        is_censored = np.int8(np.cumsum(is_censored)==0)
    
    if count_up_to:
        x = np.roll(x,1)
        x = x+1
        x[0]=init_steps_since_true 
    
    if return_censoring:
        return x,is_censored
    else: 
        return x


def steps_since_true_by_index(v,start_indx,end_indx,
                               count_up_to =False,
                               return_censoring=False) :
    """time-since-event created from a window over an array of indexes of events
        useful for large arrays or strings, 
        call v=np.where(large logical array)[0]

    Args:
        v          : 1d int array of indexes pointing out events
        start_indx : int, first index of sought window.
        end_indx   : int, last  index of sought window.
        count_up_to: If true we count up to and including step of event (TODO)

    Returns:
        steps_since: 1+end_indx-start_indx length np.int32 array
        is_censored: 1+end_indx-start_indx length np.int16 array indicating censoring

    (TODO)Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    indx_pos_before_window = np.where(np.less(v,start_indx))[0]
    indx_pos_in_window     = np.where(np.logical_and(
                                        np.greater_equal(end_indx, v), 
                                        np.greater_equal(v,start_indx)
                                        )
                                      )[0] 
    
    n = end_indx-start_indx+1 
    
    if len(indx_pos_before_window)==0&len(indx_pos_in_window)==0:
        # No events in or before window
        steps_since = np.arange(1,n+1)+start_indx
        if return_censoring:
            is_censored = np.int8(np.ones(n))
            return steps_since, is_censored
        else:
            return steps_since

    event  = np.zeros(n)
    if len(indx_pos_before_window)>0:
        # Last event-index before window 
        init_steps_since_true = start_indx-v[indx_pos_before_window][-1]
    else:
        # No events before window.
        init_steps_since_true = 1 #?

    if len(indx_pos_in_window)>0:
        event[v[indx_pos_before_window]] = 1  
    
    return steps_since_true(event,count_up_to,return_censoring,init_steps_since_true)


def steps_since_true_minimal(event):
    z = -1
    x = np.int32(event)
    for i in xrange(len(event)):
        if event[i]>0:
            z = i
        x[i] = i-z
    return x



def get_is_censored(event,count_up_to=False) :
    # (legacy)
    # Returns indicator integer of right censoring
    # If count_up_to the count is reset to 1 after step of event
    # Note that this always induces right-censoring at beginning
    event = np.int8(np.array(event)>0)
    if count_up_to :
        event = np.roll(event,1)
        event[0]=0
        
    return np.int8(np.cumsum(event)==0)
    
     
def tester_steps_since_true():
    event_sequences =np.array([
        [0,0,0,0],# 1 
        [1,0,0,0],# 2 
        [0,1,1,1],# 3 
        [0,0,1,1],# 4 
        [0,0,1,0] # 5 
        ])
    
    expected_steps_since_true = dict(
        count_up_to_false = np.array([
                [1,2,3,4],# 1 [0,0,0,0]
                [0,1,2,3],# 2 [1,0,0,0]
                [1,0,0,0],# 3 [0,1,1,1]
                [1,2,0,0],# 4 [0,0,1,1]
                [1,2,0,1] # 5 [0,0,1,0]
            ]),
        count_up_to_true =np.array([
                [1,2,3,4],# 1 [0,0,0,0]
                [1,1,2,3],# 2 [1,0,0,0]
                [1,2,1,1],# 3 [0,1,1,1]
                [1,2,3,1],# 4 [0,0,1,1]
                [1,2,3,1] # 5 [0,0,1,0]
            ])
        )
    expected_is_censored = dict(
        count_up_to_false = np.array([
                [1,1,1,1],# 1 [0,0,0,0]
                [0,0,0,0],# 2 [1,0,0,0]
                [1,0,0,0],# 3 [0,1,1,1]
                [1,1,0,0],# 4 [0,0,1,1]
                [1,1,0,0] # 5 [0,0,1,0]
            ]),
        count_up_to_true =np.array([
                [1,1,1,1],# 1 [0,0,0,0]
                [1,0,0,0],# 2 [1,0,0,0]
                [1,1,0,0],# 3 [0,1,1,1]
                [1,1,1,0],# 4 [0,0,1,1]
                [1,1,1,0] # 5 [0,0,1,0]
            ])
        )

    print '---------steps_since_true--------'
    for test_indx in xrange(2):
        count_up_to=test_indx==0
        key_in_awful_test_dict = ['count_up_to_false','count_up_to_false'][test_indx]
        
        print "count_up_to =", count_up_to
        for k in xrange(5):
            x = np.copy(event_sequences[k])
            steps_since, is_censored = steps_since_true(x,  count_up_to, return_censoring = True);
            
            expected_sst = expected_steps_since_true[key_in_awful_test_dict][k]
            expected_censoring = expected_is_censored[key_in_awful_test_dict][k]
            
            if any(expected_censoring!= is_censored) | any(expected_sst!=steps_since):
                print 'sequence  :',x,'\n',\
                  'result    :',steps_since,'\n',\
                  'expected  :',expected_sst, '\n',\
                  'cens_reslt:',is_censored, '\n',\
                  'cens_expct:',expected_censoring, '\n',k


    # for i in xrange(4):
    #     res = steps_since_true_minimal(event_sequences[i,:])-expected_steps_since_true['count_up_to_false'][i,:]
    #     if any(res!=0):
    #         print 'a'
            
#tester_steps_since_true()

def roll_fun(x,size,fun=np.mean):
    y = np.copy(x)
    n = len(x)
    size = min(size,n)
    
    if size<=1:
        return x

    for i in xrange(size):
        y[i] = fun(x[0:(i+1)])        
    for i in xrange(size,n):
        y[i] = fun(x[(i-size+1):(i+1)])
    return y

def na_locf(x):
    v = np.isnan(x)
    real_val = 0
    for i in xrange(len(x)):
        if v[i]:
            x[i] = x[real_val]
        else:
            real_val = i
    return x