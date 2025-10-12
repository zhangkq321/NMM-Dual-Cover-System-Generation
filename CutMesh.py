import numpy as np
from collections import defaultdict
from itertools import chain
import pyvista as pv
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split, unary_union
import pandas as pd

# 寻找每一个节点所在的单元
def get_elements_per_node(elements, target_node_indices):
    # 将elements转换为NumPy数组
    elements_array = np.asarray(elements)
    num_elements = elements_array.shape[0]

    # 构建节点到单元索引的映射
    node_to_elements = defaultdict(list)
    for elem_idx in range(num_elements):
        for node in elements_array[elem_idx]:
            node_to_elements[node].append(elem_idx)

    # 将列表转换为NumPy数组
    for node in node_to_elements:
        node_to_elements[node] = np.array(node_to_elements[node], dtype=np.int64)

    # 为每个目标节点提取结果
    result = []
    for node in target_node_indices:
        result.append(node_to_elements.get(node, np.array([], dtype=np.int64)))

    return result

# 线段交点计算
def is_line_intersect_joints(p1, p2, joints):

    def segments_intersect(a1, a2, b1, b2):
        def on_segment(p, q, r):
            if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
                return True
            return False

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        o1 = orientation(a1, a2, b1)
        o2 = orientation(a1, a2, b2)
        o3 = orientation(b1, b2, a1)
        o4 = orientation(b1, b2, a2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(a1, b1, a2):
            return True
        if o2 == 0 and on_segment(a1, b2, a2):
            return True
        if o3 == 0 and on_segment(b1, a1, b2):
            return True
        if o4 == 0 and on_segment(b1, a2, b2):
            return True

        return False

    for joint in joints:
        j_p1 = np.array([joint[0], joint[1]])
        j_p2 = np.array([joint[2], joint[3]])
        if segments_intersect(p1, p2, j_p1, j_p2):
            return True
    return False

# 离散线段
def discretize_segment(p1, p2, dd):
    """
    将两点之间的线段按固定距离 dd 离散化为有限个点。
    
    参数:
    - p1, p2: 二维点，形状为 (2,) 的 NumPy 数组、列表或元组。
    - dd: 每个点之间的固定距离。
    
    返回:
    - points: 形状为 (n, 2) 的 NumPy 数组，表示线段上的离散点。
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    
    delta = p2 - p1
    L = np.linalg.norm(delta)
    
    # 处理两点重合的情况
    if np.isclose(L, 0):
        return p1.reshape(1, -1)
    
    unit_vec = delta / L  # 单位方向向量
    
    # 生成 t 值（沿线段的距离）
    t_values = np.arange(0, L + 1e-8, dd)
    t_values = t_values[t_values <= L]  # 确保不超过线段长度

    # 计算每个点
    points = p1 + t_values.reshape(-1, 1) * unit_vec

    # 确保终点也被包含（如果最后一点不是终点）
    if not np.allclose(points[-1], p2, atol=1e-8):
        points = np.vstack([points, p2])

    return points

def FindBC(BC, nodes):
    BCdict = {}
    # 处理约束
    for i, bbcc in enumerate(BC):
        col_data = nodes[:, int(bbcc[0])]
        MorM = int(bbcc[1])
        if MorM == 0:
            mm = col_data.min()
        elif MorM == 1:
            mm = col_data.max()
        # 节点索引
        row_indices = np.flatnonzero(col_data == mm)
        xyz = int(bbcc[2])
        if xyz == 0 or xyz == 1:
            dofs = [xyz]
        elif xyz == 2:
            dofs = [0, 1]

        BCdict[i] = {'Index':row_indices, 'dof':dofs, 'val':bbcc[3]}
        
    return BCdict

def remove_unused_nodes(nodes, elements, elements2):
    """
    删除未使用的节点，并更新 elements 和 elements2 的索引。
    """
    used_indices = np.unique(elements)
    new_nodes = nodes[used_indices]
    
    # 定义通用的索引映射函数
    def map_indices(old_elements):
        flat = old_elements.flatten()
        new_flat = np.searchsorted(used_indices, flat)
        return new_flat.reshape(old_elements.shape)
    
    # 注意elements2的一些节点可能已经被删除，这将导致其索引错误，但是存在这样节点的单元并不会被使用，因此这里可以忽略这一问题
    new_elements = map_indices(elements)
    new_elements2 = map_indices(elements2)

    return new_nodes, new_elements, new_elements2

from dataclasses import dataclass
from typing import Optional

@dataclass
class Element:
    points: np.ndarray # 单元顶点
    gxyz: np.ndarray # 单元形心
    inxy: np.ndarray # 单元内部一点
    Area: float # 面积
    material: np.ndarray # 材料矩阵
    stress: np.ndarray # 单元应力
    dof: Optional[np.ndarray] = None # 星点索引
    PointLoad: Optional[np.ndarray] = None # 点荷载
    
# 主函数
def CutMEMesh(nodes, elements, crack, dd, materials, BC, NC, MP, polyHole):

    if polyHole is not None:
        isHole = True
    else:
        isHole = False

    # 建立pyvista格式网格grid
    num_elements = elements.shape[0]
    cells = np.empty((num_elements, 4), dtype=int)
    cells[:, 0] = 3  # 每个单元有 3 个节点
    cells[:, 1:] = elements
    celltypes = np.full(num_elements, pv.CellType.TRIANGLE, dtype=np.uint8)
    points = np.column_stack((nodes, np.zeros(nodes.shape[0])))
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    # 根据更新的crack离散为点检测所在单元
    result = []
    for i, jb in enumerate(crack):
        bb1 = np.array([jb[0], jb[1]])
        bb2 = np.array([jb[2], jb[3]])
        # 生成离散点
        point = discretize_segment(bb1, bb2, dd)
        zeros = np.zeros((point.shape[0], 1))
        point_ = np.hstack((point, zeros))
        # 检查所在的网格
        cell_indices = grid.find_containing_cell(point_)
        elementsIndex = np.unique(cell_indices)
        for ii in elementsIndex:
            result.append([i, ii])
    result = np.array(result)

    # 建立E2C:{key(elementIndex):[crackIndex]}
    E2C = defaultdict(list)
    # 遍历每一行
    for row in result:
        key = row[1]   # 第 1 列作为键
        value = row[0] # 第 0 列作为值
        E2C[key].append(value)
    # 转换为普通字典
    E2C = dict(E2C)

    ME = []
    CarckElementIndex = []
    for ele, ci in E2C.items():

        # 计算每一个单元ele被crack-ci切割后的多边形
        triangle = Polygon(nodes[elements[ele],:])
        lines = []
        for i in ci:
            lines.append(LineString(crack[i,:].reshape(2, 2)))
        # 合并切割线段（处理线段之间的交点）
        cut_lines = unary_union(lines)
        # 使用合并后的线段分割三角形
        result1 = split(triangle, cut_lines)
        # 提取分割后的多边形
        polygons = [p for p in result1.geoms if isinstance(p, Polygon)]
        # 过滤掉面积小于1e-15的多边形
        filtered_polygons = [p for p in polygons if p.area >= 1e-15]

        # 如果过滤后仅剩一个有效多边形，跳过当前循环
        if len(filtered_polygons) < 2:
            continue

        for i, poly in enumerate(polygons):
            
            # 计算多边形内部一点
            innerpoint = poly.representative_point()
            cx, cy = poly.centroid.x, poly.centroid.y

            x, y = poly.exterior.xy
            coords = np.column_stack((x, y))
            df = pd.DataFrame(coords)
            coords = df.round(6).drop_duplicates().values
            coords = np.flipud(coords)

            if isHole:
                pointhole = Point(np.array([innerpoint.x, innerpoint.y]))
                if pointhole.within(polyHole):
                    continue
            
            CarckElementIndex.append(ele)

            ME.append(Element(
                points = coords,
                gxyz = np.array([cx, cy]),
                inxy = np.array([innerpoint.x, innerpoint.y]),
                Area = poly.area,
                material = materials,
                stress = np.zeros((3)) ))

    # 被切割单元索引
    CarckElementIndex = np.array(CarckElementIndex)

    # 非切割单元矩阵
    all_indices = np.arange(elements.shape[0])
    not_in_mask = ~np.isin(all_indices, CarckElementIndex)
    NoCarckIndex = all_indices[not_in_mask]
    NoCarckElement = elements[NoCarckIndex]

    # 计算形心
    NoCarckCenterCoords = np.mean(nodes[NoCarckElement], axis=1)
    # 计算面积
    # 提取三角形各顶点坐标
    v1 = nodes[NoCarckElement[:, 0]]  # 第一个顶点
    v2 = nodes[NoCarckElement[:, 1]]  # 第二个顶点
    v3 = nodes[NoCarckElement[:, 2]]  # 第三个顶点
    # 使用向量叉乘计算面积：面积 = 0.5 * | (v2 - v1) × (v3 - v1) |
    area_vectors = np.cross(v2 - v1, v3 - v1)
    NoCarckEArea = 0.5 * np.abs(area_vectors)

    NoCarckIndex_new = []
    for i, index in enumerate(NoCarckIndex):
        if isHole:
            pointhole = Point(NoCarckCenterCoords[i,:])
            if pointhole.within(polyHole):
                continue
        
        NoCarckIndex_new.append(index)

        ME.append(Element(
            points = nodes[elements[index],:],
            gxyz = NoCarckCenterCoords[i,:],
            inxy = NoCarckCenterCoords[i,:],
            Area = NoCarckEArea[i],
            material = materials,
            stress = np.zeros((3)) ))

    NoCarckIndex = NoCarckIndex_new
    # 构建MP单元矩阵
    indexall = np.hstack((CarckElementIndex, NoCarckIndex)).astype(int)
    MPE = elements[indexall]

    if isHole:
        nodes, MPE, elements = remove_unused_nodes(nodes, MPE, elements)

    # 未被切割直接返回
    if len(CarckElementIndex) == 0:
        # 分配自由度
        for i, me in enumerate(ME):
            me.dof = MPE[i]    
        # 处理约束
        BCdict = FindBC(BC, nodes)
        # 寻找荷载点所在单元
        if len(NC) != 0:
            zeros = np.zeros((NC.shape[0], 1))
            point_ = np.hstack((NC[:,0:2], zeros))
            # 检查所在的网格
            cell_indices = grid.find_containing_cell(point_)
            for i, index in enumerate(cell_indices):
                ME[index].PointLoad = NC[i,:]
        # 寻找测量点所在单元
        if len(MP) != 0:
            zeros = np.zeros((MP.shape[0], 1))
            point_ = np.hstack((MP, zeros))
            # 检查所在的网格
            cell_indices = grid.find_containing_cell(point_)
            MPL = np.hstack((MP, cell_indices.reshape(-1,1)))
        else:
            MPL = []

        return nodes, ME, BCdict, MPL, MPE

    # 被切割单元矩阵
    CarckElement = elements[CarckElementIndex]
    # 找到所有被切割单元对应节点，即星点
    CStarNode = np.unique(CarckElement)

    # 建立数学覆盖{CStarNode}:[elementIndex]
    StarN2E = get_elements_per_node(elements, CStarNode)

    for nodeIndex, elements_list in zip(CStarNode, StarN2E):

        # 找到被所有单元中的切割单元对应的crack索引cindex
        mapped_values = [E2C[x] for x in elements_list if x in E2C]
        mapped_values_flat = list(chain.from_iterable(mapped_values))
        cindex = np.unique(mapped_values_flat)

        # 找到所有的流形单元
        mask = np.where(np.isin(indexall, elements_list))[0]
        mpoint = []
        for i in mask:
            mpoint.append(ME[i].inxy)
        mpoint = np.array(mpoint)

        # 单元边界点，用于辅助分析（避免未完全切割数学覆盖）
        DnodeIndex = np.unique(elements[elements_list])
        DnodeIndex = DnodeIndex[DnodeIndex != nodeIndex]
        Dnodes = nodes[DnodeIndex,:]

        # 合并原始节点和辅助节点
        all_points = np.vstack((mpoint, Dnodes))
        n_original = len(mpoint)  # 原始节点数量

        # 进行连通域分析（BFS算法）
        unvisited = set(range(len(all_points)))
        groups = []

        while unvisited:
            start = next(iter(unvisited))
            unvisited.remove(start)
            group = [start]
            queue = [start]

            while queue:
                current = queue.pop(0)
                for v in list(unvisited):
                    p_current = all_points[current]
                    p_v = all_points[v]
                    if not is_line_intersect_joints(p_current, p_v, crack[cindex,:]):
                        group.append(v)
                        unvisited.remove(v)
                        queue.append(v)
            groups.append(group)

        # 过滤辅助节点，仅保留原始节点的分组
        original_groups = []
        for group in groups:
            # 提取组内原始节点（索引 < n_original）
            orig_group = [idx for idx in group if idx < n_original]
            if orig_group:  # 忽略空组
                original_groups.append(orig_group)

        # 仅当存在多个分组时才需要复制节点
        if len(original_groups) > 1:
            # 第一个分组保留原节点，后续分组创建新节点
            for i, group in enumerate(original_groups[1:], start=1):
                # 复制节点
                new_node = nodes[nodeIndex, :]
                nodes = np.vstack((nodes, new_node))
                new_node_index = len(nodes) - 1  # 新节点索引
                
                # 修改原始单元矩阵中的节点索引
                for idx in group:
                    element_idx = mask[idx]  # 映射回原始单元索引
                    MPE[element_idx, MPE[element_idx] == nodeIndex] = new_node_index

    # 分配自由度
    for i, me in enumerate(ME):
        me.dof = MPE[i]

    # 建立pyvista格式网格grid
    num_elements = MPE.shape[0]
    cells = np.empty((num_elements, 4), dtype=int)
    cells[:, 0] = 3  # 每个单元有 3 个节点
    cells[:, 1:] = MPE
    celltypes = np.full(num_elements, pv.CellType.TRIANGLE, dtype=np.uint8)
    points = np.column_stack((nodes, np.zeros(nodes.shape[0])))
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    # 处理约束
    BCdict = FindBC(BC, nodes)

    # 注意这里直接采用MC寻找可能不准确，需要确认
    # 寻找荷载点所在单元
    if len(NC) != 0:
        zeros = np.zeros((NC.shape[0], 1))
        point_ = np.hstack((NC[:,0:2], zeros))
        # 检查所在的网格
        cell_indices = grid.find_containing_cell(point_)
        for i, index in enumerate(cell_indices):
            ME[index].PointLoad = NC[i,:]

    # 寻找测量点所在单元
    if len(MP) != 0:
        zeros = np.zeros((MP.shape[0], 1))
        point_ = np.hstack((MP, zeros))
        # 检查所在的网格
        cell_indices = grid.find_containing_cell(point_)
        MPL = np.hstack((MP, cell_indices.reshape(-1,1)))
    else:
        MPL = []      
        
    return nodes, ME, BCdict, MPL, MPE