import numpy as np
arr=np.array([[5,6,4],[2,5,6],[3,5,6]])
u, s, vt =np.linalg.svd(arr)
print("U matrix\n",u)
print(np.diag(s))
print(vt)
reconstruct_arr=np.dot(u,np.dot(np.diag(s),vt))
print(reconstruct_arr)