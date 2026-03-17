# import multiprocessing
import numpy as np
import pickle

class Dataset_vvtk(object):
   
    def __init__(self, file_data, mode='rb', debug=True, index=None):
        self.file_data = file_data
        self.debug = debug
        self.mode = mode
        print('  file_data: %s\n' % self.file_data)
        self.f = open(self.file_data, mode)
        
        self.dtype2int = {'uint8': 0, 'int8': 1, 'int16': 2, 'int32': 3, 'int64': 4, 'float32': 5, 'float64': 6}
        self.int2dtype =  {0: 'uint8', 1: 'int8', 2: 'int16', 3: 'int32', 4: 'int64', 5: 'float32', 6: 'float64'}
        self.int2size = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 4, 6: 8}
        
        if index is not None:
            self.index = index
        else:
            self.index = {}
            if mode == 'wb':
                self.f.write( np.array([0], dtype=np.int64).tobytes() ) 
            else:
                pos = np.frombuffer(self.f.read(8), dtype=np.int64)[0]
                self.f.seek(pos)
                self.index = pickle.load(self.f)
                # self.lock = multiprocessing.Lock()
        
                self.f.close()
                                         
    
    def get(self, file, index=None):
        index = self.index if index is None else index                                                               
        # self.lock.acquire()     
        self.f = open(self.file_data, 'rb')                                                       
        self.f.seek(index[file])                                                  
        type = np.frombuffer(self.f.read(1), dtype=np.int8)[0]                         
        dim = np.frombuffer(self.f.read(1), dtype=np.int8)[0]                          
        shape = np.frombuffer(self.f.read(4*dim), dtype=np.int32)                      
        x = np.frombuffer(self.f.read(self.int2size[type]*shape.prod()), dtype=self.int2dtype[type])                                                                          
        # self.lock.release()  
        self.f.close()                                                          
        x = np.copy(x)                                                                 
        return x.reshape(*shape)                                                       

    
    
    def add(self, x, file):
        self.index[file] = self.f.tell()
        self.f.write( np.array([self.dtype2int[str(x.dtype)]], dtype=np.int8).tobytes() ) 
        self.f.write( np.array([len(x.shape)], dtype=np.int8).tobytes() ) 
        self.f.write( np.array(x.shape, dtype=np.int32).tobytes() )
        self.f.write( x.tobytes() )

    def close(self):
        if self.mode!='r':
            pos  = self.f.tell()  # postion after files            
            pickle.dump(self.index, self.f, protocol=2)
            self.f.seek(0)
            self.f.write( np.array([pos], dtype=np.int64).tobytes() ) 
        
        self.f.close()

if __name__ == '__main__':
     
    dataset = Dataset_vvtk('data', 'wb')
    x = (np.random.randn(10,) * 5).astype(np.int16)
    print('x0', x)
    dataset.add(x, 'file1')
    dataset.close()
    
    dataset = Dataset_vvtk('data', 'rb')
    
    x1 = dataset.get('file1')
    print('x1', x1)
    
    import os
    os.remove('data')
    
    