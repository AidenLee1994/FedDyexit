import signal
import sys


class Signal(object):
    def __init__(self):
        signal.signal(signal.SIGINT,self.signal_handler)
        
    def signal_handler(self,signal, frame):
        print("End running! Press Ctrl+C!")
        sys.exit()
        
    
if __name__ == "__main__":
    Signal()
    while(True):
        print("1")