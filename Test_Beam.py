import numpy as np
from ShapeF import GetFEMesh
from CutMesh import CutMEMesh
from AddMat import AddMat, GetS2ME
from DrawAll import PlotMesh, PoltCountor
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve  #稀疏矩阵求解器

polygon = [
    [0, 0],
    [20, 0],
    [20, 2],
    [0, 2]
]

crack = np.array([[0, 1, 20, 1]]).astype(float)

meshsize = 0.8

material = np.array([10000, 0.2, 100]) # E, v, den

# 约束面 xy->01,minmax->01,x/y/xy->012,val
BC = np.array([[0, 0, 2, 0]])

# 荷载点 x,y,fx,fy
NC = np.array([[20, 2, 0, 1]])

# 测量点 x, y
MP = np.array([[20, 2]])

###################################################
# 前处理
# 生成网格
nodes_origin, elements = GetFEMesh(polygon, meshsize)

# 生成流形单元
nodes, ME, BCdict, MPL, MPE = CutMEMesh(nodes_origin, elements, crack, meshsize/10, material, BC, NC, MP, None)

# 网格图
PlotMesh(crack, ME, nodes, MPE)

# 静力矩阵
K, F = AddMat(nodes, ME, BCdict)

# 求解
K = csr_matrix(K)
Dnew = spsolve(K, F) 

# 后处理
ME, uu, us, MPu = GetS2ME(nodes, ME, Dnew, MPL)

# 云图
PoltCountor(ME, uu[:,1])
