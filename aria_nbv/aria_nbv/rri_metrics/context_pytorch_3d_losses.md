pytorch3d.loss
Loss functions for meshes and point clouds.

pytorch3d.loss.chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, batch_reduction: str | None = 'mean', point_reduction: str | None = 'mean', norm: int = 2, single_directional: bool = False, abs_cosine: bool = True)[source]
Chamfer distance between two pointclouds x and y.

Parameters
:
x – FloatTensor of shape (N, P1, D) or a Pointclouds object representing a batch of point clouds with at most P1 points in each batch element, batch size N and feature dimension D.

y – FloatTensor of shape (N, P2, D) or a Pointclouds object representing a batch of point clouds with at most P2 points in each batch element, batch size N and feature dimension D.

x_lengths – Optional LongTensor of shape (N,) giving the number of points in each cloud in x.

y_lengths – Optional LongTensor of shape (N,) giving the number of points in each cloud in y.

x_normals – Optional FloatTensor of shape (N, P1, D).

y_normals – Optional FloatTensor of shape (N, P2, D).

weights – Optional FloatTensor of shape (N,) giving weights for batch elements for reduction operation.

batch_reduction – Reduction operation to apply for the loss across the batch, can be one of [“mean”, “sum”] or None.

point_reduction – Reduction operation to apply for the loss across the points, can be one of [“mean”, “sum”, “max”] or None. Using “max” leads to the Hausdorff distance.

norm – int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

single_directional – If False (default), loss comes from both the distance between each point in x and its nearest neighbor in y and each point in y and its nearest neighbor in x. If True, loss is the distance between each point in x and its nearest neighbor in y.

abs_cosine – If False, loss_normals is from one minus the cosine similarity. If True (default), loss_normals is from one minus the absolute value of the cosine similarity, which means that exactly opposite normals are considered equivalent to exactly matching normals, i.e. sign does not matter.

Returns
:
2-element tuple containing

loss: Tensor giving the reduced distance between the pointclouds in x and the pointclouds in y. If point_reduction is None, a 2-element tuple of Tensors containing forward and backward loss terms shaped (N, P1) and (N, P2) (if single_directional is False) or a Tensor containing loss terms shaped (N, P1) (if single_directional is True) is returned.

loss_normals: Tensor giving the reduced cosine distance of normals between pointclouds in x and pointclouds in y. Returns None if x_normals and y_normals are None. If point_reduction is None, a 2-element tuple of Tensors containing forward and backward loss terms shaped (N, P1) and (N, P2) (if single_directional is False) or a Tensor containing loss terms shaped (N, P1) (if single_directional is True) is returned.

pytorch3d.loss.mesh_edge_loss(meshes, target_length: float = 0.0)[source]
Computes mesh edge length regularization loss averaged across all meshes in a batch. Each mesh contributes equally to the final loss, regardless of the number of edges per mesh in the batch by weighting each mesh with the inverse number of edges. For example, if mesh 3 (out of N) has only E=4 edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to contribute to the final loss.

Parameters
:
meshes – Meshes object with a batch of meshes.

target_length – Resting value for the edge length.

Returns
:
loss – Average loss across the batch. Returns 0 if meshes contains no meshes or all empty meshes.

pytorch3d.loss.mesh_laplacian_smoothing(meshes, method: str = 'uniform')[source]
Computes the laplacian smoothing objective for a batch of meshes. This function supports three variants of Laplacian smoothing, namely with uniform weights(“uniform”), with cotangent weights (“cot”), and cotangent curvature (“cotcurv”).For more details read [1, 2].

Parameters
:
meshes – Meshes object with a batch of meshes.

method – str specifying the method for the laplacian.

Returns
:
loss – Average laplacian smoothing loss across the batch. Returns 0 if meshes contains no meshes or all empty meshes.

Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3. The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors: for a uniform Laplacian, LuV[i] points to the centroid of its neighboring vertices, a cotangent Laplacian LcV[i] is known to be an approximation of the surface normal, while the curvature variant LckV[i] scales the normals by the discrete mean curvature. For vertex i, assume S[i] is the set of neighboring vertices to i, a_ij and b_ij are the “outside” angles in the two triangles connecting vertex v_i and its neighboring vertex v_j for j in S[i], as seen in the diagram below.

       a_ij
        /\
       /  \
      /    \
     /      \
v_i /________\ v_j
    \        /
     \      /
      \    /
       \  /
        \/
       b_ij

The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
For the uniform variant,    w_ij = 1 / |S[i]|
For the cotangent variant,
    w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
where A[i] is the sum of the areas of all triangles containing vertex v_i.
There is a nice trigonometry identity to compute cotangents. Consider a triangle with side lengths A, B, C and angles a, b, c.

       c
      /|\
     / | \
    /  |  \
 B /  H|   \ A
  /    |    \
 /     |     \
/a_____|_____b\
       C

Then cot a = (B^2 + C^2 - A^2) / 4 * area
We know that area = CH/2, and by the law of cosines we have

A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

Putting these together, we get:

B^2 + C^2 - A^2     2BC cos a
_______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
   4 * area            2CH
[1] Desbrun et al, “Implicit fairing of irregular meshes using diffusion and curvature flow”, SIGGRAPH 1999.

[2] Nealan et al, “Laplacian Mesh Optimization”, Graphite 2006.

pytorch3d.loss.mesh_normal_consistency(meshes)[source]
Computes the normal consistency of each mesh in meshes. We compute the normal consistency for each pair of neighboring faces. If e = (v0, v1) is the connecting edge of two neighboring faces f0 and f1, then the normal consistency between f0 and f1

        a
        /\
       /  \
      / f0 \
     /      \
v0  /____e___\ v1
    \        /
     \      /
      \ f1 /
       \  /
        \/
        b
The normal consistency is

nc(f0, f1) = 1 - cos(n0, n1)

where cos(n0, n1) = n0^n1 / ||n0|| / ||n1|| is the cosine of the angle
between the normals n0 and n1, and

n0 = (v1 - v0) x (a - v0)
n1 = - (v1 - v0) x (b - v0) = (b - v0) x (v1 - v0)
This means that if nc(f0, f1) = 0 then n0 and n1 point to the same direction, while if nc(f0, f1) = 2 then n0 and n1 point opposite direction.

Note

For well-constructed meshes the assumption that only two faces share an edge is true. This assumption could make the implementation easier and faster. This implementation does not follow this assumption. All the faces sharing e, which can be any in number, are discovered.

Parameters
:
meshes – Meshes object with a batch of meshes.

Returns
:
loss – Average normal consistency across the batch. Returns 0 if meshes contains no meshes or all empty meshes.

pytorch3d.loss.point_mesh_edge_distance(meshes: Meshes, pcls: Pointclouds)[source]
Computes the distance between a pointcloud and a mesh within a batch. Given a pair (mesh, pcl) in the batch, we define the distance to be the sum of two distances, namely point_edge(mesh, pcl) + edge_point(mesh, pcl)

point_edge(mesh, pcl): Computes the squared distance of each point p in pcl
to the closest edge segment in mesh and averages across all points in pcl

edge_point(mesh, pcl): Computes the squared distance of each edge segment in mesh
to the closest point in pcl and averages across all edges in mesh.

The above distance functions are applied for all (mesh, pcl) pairs in the batch and then averaged across the batch.

Parameters
:
meshes – A Meshes data structure containing N meshes

pcls – A Pointclouds data structure containing N pointclouds

Returns
:
loss –

The point_edge(mesh, pcl) + edge_point(mesh, pcl) distance
between all (mesh, pcl) in a batch averaged across the batch.

pytorch3d.loss.point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds, min_triangle_area: float = 0.005)[source]
Computes the distance between a pointcloud and a mesh within a batch. Given a pair (mesh, pcl) in the batch, we define the distance to be the sum of two distances, namely point_face(mesh, pcl) + face_point(mesh, pcl)

point_face(mesh, pcl): Computes the squared distance of each point p in pcl
to the closest triangular face in mesh and averages across all points in pcl

face_point(mesh, pcl): Computes the squared distance of each triangular face in
mesh to the closest point in pcl and averages across all faces in mesh.

The above distance functions are applied for all (mesh, pcl) pairs in the batch and then averaged across the batch.

Parameters
:
meshes – A Meshes data structure containing N meshes

pcls – A Pointclouds data structure containing N pointclouds

min_triangle_area – (float, defaulted) Triangles of area less than this will be treated as points/lines.

Returns
:
loss –

The point_face(mesh, pcl) + face_point(mesh, pcl) distance
between all (mesh, pcl) in a batch averaged across the batch.