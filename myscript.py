import numpy as np
toto=np.random.rand(10,10)
print(str(np.shape(toto)))
toto_mean=np.mean(toto)
toto_std=np.std(toto)
np.save('toto.npy', toto)
print('MEAN=%f, STD=%f' %(toto_mean, toto_std))