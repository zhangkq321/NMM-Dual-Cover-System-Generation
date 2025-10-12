import numpy as np
from ShapeF import GetFEMesh
from CutMesh import CutMEMesh
from AddMat import AddMat, GetS2ME
from DrawAll import PlotMesh, PoltCountor
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve  #稀疏矩阵求解器

# 将一个圆离散成一系列直线段
# input 圆心坐标[X,Y]，半径，离散线段数，角度
def discretize_circle(center, radius, num_lines, degrees):

    radius = radius

    curve_number = (degrees * np.pi) / 180

    circle_points = np.empty((0, 2))
    angle_increment = curve_number / num_lines

    for i in range(num_lines):
        angle = i * angle_increment
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        circle_points = np.append(circle_points, [[x, y]], axis=0)

    if degrees == 360:
        circle_points = np.append(circle_points, [[circle_points[0][0], circle_points[0][1]]], axis=0)

    return circle_points

# 外边界
def out_segment(outer_points):
    # 创建一个空列表用于存储边的起点和终点坐标，平铺为一维
    edges_flattened = []

    # 遍历所有顶点，构建边
    for i in range(len(outer_points)):
        # 每条边由当前顶点和下一个顶点组成
        edge_start = outer_points[i]
        # 如果是最后一个顶点，则连接到第一个顶点，形成闭合多边形
        edge_end = outer_points[(i + 1) % len(outer_points)]
        # 将起点和终点坐标平铺后添加到列表中
        edges_flattened.append([edge_start[0], edge_start[1], edge_end[0], edge_end[1]])

    # 将列表转换为numpy数组
    edges_array = np.array(edges_flattened)
    return np.delete(edges_array, -1, axis=0)

#################################################################
polygon = [
    [0, 0],
    [20, 0],
    [20, 20],
    [0, 20]
]

circle_points = discretize_circle([10,10], 3, 30, 360)
crack1 = out_segment(circle_points)
crack2 = np.array([3, 10, 18, 10])

crack = np.vstack((crack1, crack2))

from shapely import Polygon, MultiPolygon
poly = Polygon(circle_points)

meshsize = 0.8

material = np.array([1500000, 0.3, 100]) # E, v, den

# 约束面 xy->01,minmax->01,x/y/xy->012,val
BC = np.array([[1, 0, 1, -1],
               [1, 1, 1, 1]])

# 荷载点 x,y,fx,fy
NC = np.array([])

# 测量点 x, y
MP = np.array([])

###################################################
# 前处理
# 生成网格
nodes_origin, elements = GetFEMesh(polygon, meshsize)

# 生成流形单元
nodes, ME, BCdict, MPL, MPE = CutMEMesh(nodes_origin, elements, crack, meshsize/10, material, BC, NC, MP, poly)

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
