import numpy as np

def triangle_area(v1, v2, v3):
    v1v2 = v2 - v1
    v1v3 = v3 - v1

    cross_product = v1v2[0] * v1v3[1] - v1v2[1] * v1v3[0]

    area = abs(cross_product) / 2.0
    return area

def lagrange_basis(LocalCoord):
    """
    计算三节点三角形单元的形函数及其导数
    参数:
        xi, eta: 自然坐标（面积坐标）
    返回:
        N: 形函数数组 [N1, N2, N3]
        dN: 形函数对自然坐标的导数 (3x2矩阵)
    """

    xi, eta = LocalCoord

    # 计算形函数
    N1 = 1 - xi - eta
    N2 = xi
    N3 = eta
    N = np.array([N1, N2, N3])
    
    # 形函数对自然坐标的导数
    dN = np.array([
        [-1, -1],  # dN1/dxi, dN1/deta
        [ 1,  0],  # dN2/dxi, dN2/deta
        [ 0,  1]   # dN3/dxi, dN3/deta
    ])
    
    return N, dN

def NFunction2D(gxy, nodes):
    """
    在全局坐标下计算三角形单元的重心坐标形函数
    """

    # 解包节点坐标
    A = nodes[0]
    B = nodes[1]
    C = nodes[2]
    
    Area_ABC = triangle_area(A, B, C)
    Area_PCA = triangle_area(gxy, C, A)
    Area_PAB = triangle_area(gxy, A, B)
    
    LocalCoord = np.zeros(2)
    LocalCoord[0] = Area_PCA / Area_ABC
    LocalCoord[1] = Area_PAB / Area_ABC

    N1D, _ = lagrange_basis(LocalCoord)

    N = np.kron(N1D[np.newaxis], np.eye(2))
    
    return N

def strain_matrix(gxy, nodes):
    """
    计算三节点三角形单元的应变矩阵B
    """
    # 解包节点坐标
    A = nodes[0]
    B = nodes[1]
    C = nodes[2]
    
    Area_ABC = triangle_area(A, B, C)
    Area_PCA = triangle_area(gxy, C, A)
    Area_PAB = triangle_area(gxy, A, B)
    
    LocalCoord = np.zeros(2)
    LocalCoord[0] = Area_PCA / Area_ABC
    LocalCoord[1] = Area_PAB / Area_ABC

    _, dNdxi = lagrange_basis(LocalCoord)

    J = nodes.T @ dNdxi  # shape: (2, 2)

    dNdx = dNdxi @ np.linalg.inv(J)

    num_nodes = nodes.shape[0]

    B = np.zeros((3, 2 * num_nodes))

    for inode in range(num_nodes):
        i = inode
        dNdxi_i = dNdx[i, 0]
        dNdxj_i = dNdx[i, 1]
        
        B[:, 2*i : 2*i + 2] = np.array([
            [dNdxi_i,         0       ],
            [0,               dNdxj_i ],
            [dNdxj_i,         dNdxi_i ]
        ])
    
    return B

def Elastic(E, poisson):

    # Plane Strain state

    Daux1 = ( E * (1. - poisson) )/( ( 1.+poisson )*( 1.-2.*poisson ) )
    Daux2 = poisson/( 1.-poisson )
    Daux3 = ( 1.-2.*poisson)/( 2.*( 1.-poisson ))

    D = np.array([ [   1. , Daux2,   0. ],
                   [ Daux2,   1. ,   0. ],
                   [   0. ,   0. , Daux3] ])

    D = Daux1 * D
    
    return D

###############################
import gmsh

# 生成网格
def GetFEMesh(polygon, mm):

    # 初始化 Gmsh
    gmsh.initialize()

    # 创建新模型
    gmsh.model.add("polygon_mesh")

    # 创建点
    points = []
    for i, (x, y) in enumerate(polygon):
        points.append(gmsh.model.geo.addPoint(x, y, 0, 1.0))  # z=0, lc=1.0（网格大小）

    # 创建线段连接点
    lines = []
    for i in range(len(points)):
        lines.append(gmsh.model.geo.addLine(points[i], points[(i+1)%len(points)]))

    # 创建线环
    curve_loop = gmsh.model.geo.addCurveLoop(lines)

    # 创建面
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # 同步几何模型
    gmsh.model.geo.synchronize()

    # 设置全局网格大小因子（可选）
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mm)  # 缩放因子为 0.5

    # 设置网格算法（2D 网格）
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D

    # 生成网格
    gmsh.model.mesh.generate(2)

    # # 启动 Gmsh GUI 查看结果
    # gmsh.fltk.run()

    # === 获取节点数据 ===
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]  # 只取 x, y

    # === 获取单元数据 ===
    _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

    # 假设只有一种单元类型（如三角形）
    elements = np.array(elem_node_tags[0]).reshape(-1, 3) - 1  # 转换为 0-based index

    # 清理并结束
    gmsh.finalize()

    return nodes, elements
