import datetime
import numpy as np

def format_time(elapsed: float):
    '''
        Function for presenting time in 
        
        Parameters
        ----------
        elapsed: time in seconds
        Returns
        -------
        elapsed as string
    '''
    return str(datetime.timedelta(seconds=int(round((elapsed)))))