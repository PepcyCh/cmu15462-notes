#include <float.h>
#include <assert.h>
#include "meshEdit.h"
#include "mutablePriorityQueue.h"
#include "error_dialog.h"

namespace CMU462 {

VertexIter HalfedgeMesh::splitEdge(EdgeIter e0) {
    // This method should split the given edge and return an iterator to the
    // newly inserted vertex. The halfedge of this vertex should point along
    // the edge that was split, rather than the new edges.

    vector<HalfedgeIter> left, right;
    vector<VertexIter> verts;

    HalfedgeIter h0 = e0->halfedge();
    HalfedgeIter h1 = h0->twin();
    FaceIter f0 = h0->face();
    FaceIter f1 = h1->face();

    right.push_back(h1);
    for (HalfedgeIter h = h1->next(); h != h1; h = h->next()) {
        right.push_back(h);
        verts.push_back(h->vertex());
    }
    left.push_back(h0);
    for (HalfedgeIter h = h0->next(); h != h0; h = h->next()) {
        left.push_back(h);
        verts.push_back(h->vertex());
    }

    VertexIter v = newVertex();
    VertexIter v0 = h0->vertex();
    VertexIter v1 = h1->vertex();
    v->position = (v0->position + v1->position) / 2;

    EdgeIter e = newEdge();
    HalfedgeIter h2 = newHalfedge();
    HalfedgeIter h3 = newHalfedge();

    h1->twin() = h2;
    h2->twin() = h1;
    h0->twin() = h3;
    h3->twin() = h0;
    h2->vertex() = v;
    h3->vertex() = v;

    h0->edge() = e;
    h1->edge() = e0;
    h2->edge() = e0;
    h3->edge() = e;
    e->halfedge() = h0;
    e0->halfedge() = h1;
    v->halfedge() = h2;

    if (!f1->isBoundary()) {
        vector<HalfedgeIter> temp_halfedges;
        vector<FaceIter> temp_faces;

        temp_halfedges.push_back(h3);
        for (int i = 1; i < right.size() - 1; i++) {
            EdgeIter ne = newEdge();
            HalfedgeIter nh0 = newHalfedge();
            HalfedgeIter nh1 = newHalfedge();
            ne->halfedge() = nh0;

            nh0->twin() = nh1;
            nh1->twin() = nh0;
            nh0->vertex() = verts[i];
            nh1->vertex() = v;
            nh0->edge() = ne;
            nh1->edge() = ne;

            temp_halfedges.push_back(nh0);
            temp_halfedges.push_back(nh1);

            FaceIter nf = newFace();
            temp_faces.push_back(nf);
        }
        temp_halfedges.push_back(h1);
        temp_faces.push_back(f1);

        for (int i = 0; i < temp_faces.size(); i++) {
            temp_faces[i]->halfedge() = temp_halfedges[i << 1];
            temp_halfedges[i << 1]->next() = right[i + 1];
            right[i + 1]->next() = temp_halfedges[i << 1 | 1];
            temp_halfedges[i << 1 | 1]->next() = temp_halfedges[i << 1];
            temp_halfedges[i << 1]->face() = temp_faces[i];
            temp_halfedges[i << 1 | 1]->face() = temp_faces[i];
            right[i + 1]->face() = temp_faces[i];
        }
    } else {
        h3->next() = h1->next();
        h1->next() = h3;
        right.back()->next() = h1;
        h3->face() = h1->face() = f1;
    }
    if (!f0->isBoundary()) {
        vector<HalfedgeIter> temp_halfedges;
        vector<FaceIter> temp_faces;

        temp_halfedges.push_back(h2);
        for (int i = 0; i < left.size() - 2; i++) {
            EdgeIter ne = newEdge();
            HalfedgeIter nh0 = newHalfedge();
            HalfedgeIter nh1 = newHalfedge();
            ne->halfedge() = nh0;

            nh0->twin() = nh1;
            nh1->twin() = nh0;
            nh0->vertex() = verts[i + right.size()];
            nh1->vertex() = v;
            nh0->edge() = ne;
            nh1->edge() = ne;

            temp_halfedges.push_back(nh0);
            temp_halfedges.push_back(nh1);

            FaceIter nf = newFace();
            temp_faces.push_back(nf);
        }
        temp_halfedges.push_back(h0);
        temp_faces.push_back(f0);

        for (int i = 0; i < temp_faces.size(); i++) {
            temp_faces[i]->halfedge() = temp_halfedges[i << 1];
            temp_halfedges[i << 1]->next() = left[i + 1];
            left[i + 1]->next() = temp_halfedges[i << 1 | 1];
            temp_halfedges[i << 1 | 1]->next() = temp_halfedges[i << 1];
            temp_halfedges[i << 1]->face() = temp_faces[i];
            temp_halfedges[i << 1 | 1]->face() = temp_faces[i];
            left[i + 1]->face() = temp_faces[i];
        }
    } else {
        h2->next() = h0->next();
        h0->next() = h2;
        left.back()->next() = h0;
        h2->face() = h0->face() = f0;
    }

    // showError("splitEdge() not implemented.");
    return v;
}

VertexIter HalfedgeMesh::collapseEdge(EdgeIter e) {
    // This method should collapse the given edge and return an iterator to
    // the new vertex created by the collapse.

    // if (e->isBoundary()) return e->halfedge()->vertex();

    HalfedgeIter h0 = e->halfedge();
    HalfedgeIter h1 = h0->twin();
    FaceIter f0 = h0->face();
    FaceIter f1 = h1->face();

    if (faces.size() - (f0->degree() == 3 ? 1 : 0) - (f1->degree() == 3 ? 1 : 0) <= 2)
        return h0->vertex();

    VertexIter v = newVertex();
    VertexIter v0 = h0->vertex();
    VertexIter v1 = h1->vertex();
    v->position = (v0->position + v1->position) / 2;

    vector<HalfedgeIter> up, down;
    for (HalfedgeIter h = h1->next(); h->twin() != h1; h = h->twin()->next()) {
        up.push_back(h);
        up.push_back(h->twin());
        h->vertex() = v;
        v->halfedge() = h;
    }
    for (HalfedgeIter h = h0->next(); h->twin() != h0; h = h->twin()->next()) {
        down.push_back(h);
        down.push_back(h->twin());
        h->vertex() = v;
        v->halfedge() = h;
    }

    up.back()->next() = down[0];
    down.back()->next() = up[0];
    h0->face()->halfedge() = down[0];
    h1->face()->halfedge() = up[0];

    {
        vector<HalfedgeIter> temp;
        HalfedgeIter h = h1->next();
        do {
            temp.push_back(h);
            h = h->next();
        } while (h != up[0]);

        if (temp.size() == 2) {
            VertexIter vl = temp[1]->vertex();
            HalfedgeIter hl = temp[1]->twin()->next();
            temp[0]->next() = hl;
            vl->halfedge() = hl;

            HalfedgeIter hn = hl;
            while (hn->next() != temp[1]->twin()) hn = hn->next();
            hn->next() = temp[0];
            v->halfedge() = temp[0];

            FaceIter fo = temp[0]->face();
            temp[0]->face() = hn->face();
            hn->face()->halfedge() = hn;

            deleteFace(fo);
            deleteEdge(temp[1]->edge());
            deleteHalfedge(temp[1]->twin());
            deleteHalfedge(temp[1]);
        }
    }
    {
        vector<HalfedgeIter> temp;
        HalfedgeIter h = h0->next();
        do {
            temp.push_back(h);
            h = h->next();
        } while (h != down[0]);

        if (temp.size() == 2) {
            VertexIter vr = temp[1]->vertex();
            HalfedgeIter hr = temp[1]->twin()->next();
            temp[0]->next() = hr;
            vr->halfedge() = hr;

            HalfedgeIter hn = hr;
            while (hn->next() != temp[1]->twin()) hn = hn->next();
            hn->next() = temp[0];
            v->halfedge() = temp[0];

            FaceIter fo = temp[0]->face();
            temp[0]->face() = hn->face();
            hn->face()->halfedge() = hn;

            deleteFace(fo);
            deleteEdge(temp[1]->edge());
            deleteHalfedge(temp[1]->twin());
            deleteHalfedge(temp[1]);
        }
    }

    deleteHalfedge(h0);
    deleteHalfedge(h1);
    deleteEdge(e);
    deleteVertex(v0);
    deleteVertex(v1);

    // showError("collapseEdge() not implemented.");
    return v;
}

VertexIter HalfedgeMesh::collapseFace(FaceIter f) {
    // This method should collapse the given face and return an iterator to
    // the new vertex created by the collapse.

    if (f->isBoundary())
        return f->halfedge()->vertex();

    vector<EdgeIter> edges;
    Vector3D pos(0);
    int newFaceCnt = faces.size();
    {
        HalfedgeIter h = f->halfedge();
        do {
            edges.push_back(h->edge());
            pos += h->vertex()->position;
            newFaceCnt -= h->face()->degree() == 3;
            h = h->next();
        } while (h != f->halfedge());
    }
    pos /= edges.size();
    edges.pop_back();

    if (newFaceCnt <= 2)
        return f->halfedge()->vertex();

    VertexIter v;
    for (auto e : edges) v = collapseEdge(e);
    v->position = pos;

    // showError("collapseFace() not implemented.");
    return v;
}

FaceIter HalfedgeMesh::eraseVertex(VertexIter v) {
    // This method should replace the given vertex and all its neighboring
    // edges and faces with a single face, returning the new face.

    if (v->isBoundary())
        return v->halfedge()->face();

    if (faces.size() - v->degree() <= 2)
        return v->halfedge()->face();

    vector<HalfedgeIter> halfedges;
    vector<VertexIter> verts;
    vector<FaceIter> faces;

    HalfedgeIter th = v->halfedge();
    do {
        HalfedgeIter h;
        for (h = th->next(); h->next() != th; h = h->next()) {
            halfedges.push_back(h);
            verts.push_back(h->vertex());
        }
        faces.push_back(th->face());
        th = h->twin();
    } while (th != v->halfedge());

    for (int i = 0; i < halfedges.size(); i++) {
        verts[i]->halfedge() = halfedges[i];
        halfedges[i]->next() = halfedges[(i + 1) % halfedges.size()];
        halfedges[i]->face() = faces[0];
    }
    faces[0]->halfedge() = halfedges[0];
    for (int i = 1; i < faces.size(); i++)
        deleteFace(faces[i]);

    th = v->halfedge();
    do {
        HalfedgeIter nh = th->twin()->next();
        deleteEdge(th->edge());
        deleteHalfedge(th->twin());
        deleteHalfedge(th);
    } while (th != v->halfedge());
    deleteVertex(v);

    // showError("eraseVertex() not implemented.");
    return faces[0];
}

FaceIter HalfedgeMesh::eraseEdge(EdgeIter e) {
    // This method should erase the given edge and return an iterator to the
    // merged face.

    if (e->isBoundary())
        return e->halfedge()->face();

    vector<HalfedgeIter> halfedges;

    HalfedgeIter h0 = e->halfedge();
    HalfedgeIter h1 = h0->twin();
    FaceIter f0 = h0->face();
    FaceIter f1 = h1->face();
    VertexIter v0 = h0->vertex();
    VertexIter v1 = h1->vertex();

    if (v0->degree() + int(v0->isBoundary()) == 2 || v1->degree() + int(v1->isBoundary()) == 2)
        return f0;

    int rightSize = 0;
    for (HalfedgeIter h = h1->next(); h != h1; h = h->next()) {
        halfedges.push_back(h);
        ++rightSize;
    }
    v0->halfedge() = halfedges[0];
    for (HalfedgeIter h = h0->next(); h != h0; h = h->next()) {
        halfedges.push_back(h);
    }
    v1->halfedge() = halfedges[rightSize];

    for (int i = 0; i < halfedges.size(); i++) {
        halfedges[i]->next() = halfedges[(i + 1) % halfedges.size()];
        halfedges[i]->face() = f0;
    }
    f0->halfedge() = halfedges[0];

    deleteHalfedge(h0);
    deleteHalfedge(h1);
    deleteEdge(e);
    deleteFace(f1);

    // showError("eraseEdge() not implemented.");
    return f0;
}

EdgeIter HalfedgeMesh::flipEdge(EdgeIter e0) {
    // This method should flip the given edge and return an iterator to the
    // flipped edge.

    if (e0->isBoundary()) return e0;

    vector<HalfedgeIter> left, right, outer;
    vector<VertexIter> verts;
    vector<EdgeIter> edges;

    HalfedgeIter h0 = e0->halfedge();
    HalfedgeIter h1 = h0->twin();
    FaceIter f0 = h0->face();
    FaceIter f1 = h1->face();

    right.push_back(h1);
    for (HalfedgeIter h = h1->next(); h != h1; h = h->next()) {
        right.push_back(h);
        outer.push_back(h->twin());
        verts.push_back(h->vertex());
        edges.push_back(h->edge());
    }
    left.push_back(h0);
    for (HalfedgeIter h = h0->next(); h != h0; h = h->next()) {
        left.push_back(h);
        outer.push_back(h->twin());
        verts.push_back(h->vertex());
        edges.push_back(h->edge());
    }

    VertexIter v0 = verts[1];
    VertexIter v1 = verts[right.size()];

    {
        HalfedgeIter h = v0->halfedge();
        do {
            h = h->twin();
            if (h->vertex() == v1)
                return e0;
            h = h->next();
        } while (h != v0->halfedge());
    }

    {
        int i = 1;
        for (HalfedgeIter h = h1->next(); h != h1; h = h->next(), i++) {
            h->twin() = outer[i];
            outer[i]->twin() = h;
            h->vertex() = verts[i];
            verts[i]->halfedge() = h;
            h->face() = f1;
            f1->halfedge() = h;
            h->edge() = edges[i];
            edges[i]->halfedge() = h;
        }
        for (HalfedgeIter h = h0->next(); h != h0; h = h->next(), i = (i + 1) % outer.size()) {
            h->twin() = outer[i];
            outer[i]->twin() = h;
            h->vertex() = verts[i];
            verts[i]->halfedge() = h;
            h->face() = f0;
            f0->halfedge() = h;
            h->edge() = edges[i];
            edges[i]->halfedge() = h;
        }
    }
    h0->vertex() = v0;
    h1->vertex() = v1;

    // showError("flipEdge() not implemented.");
    return e0;
}

void HalfedgeMesh::subdivideQuad(bool useCatmullClark) {
  // Unlike the local mesh operations (like bevel or edge flip), we will perform
  // subdivision by splitting *all* faces into quads "simultaneously."  Rather
  // than operating directly on the halfedge data structure (which as you've
  // seen
  // is quite difficult to maintain!) we are going to do something a bit nicer:
  //
  //    1. Create a raw list of vertex positions and faces (rather than a full-
  //       blown halfedge mesh).
  //
  //    2. Build a new halfedge mesh from these lists, replacing the old one.
  //
  // Sometimes rebuilding a data structure from scratch is simpler (and even
  // more
  // efficient) than incrementally modifying the existing one.  These steps are
  // detailed below.

  // Step I: Compute the vertex positions for the subdivided mesh.  Here
  // we're
  // going to do something a little bit strange: since we will have one vertex
  // in
  // the subdivided mesh for each vertex, edge, and face in the original mesh,
  // we
  // can nicely store the new vertex *positions* as attributes on vertices,
  // edges,
  // and faces of the original mesh.  These positions can then be conveniently
  // copied into the new, subdivided mesh.
  if (useCatmullClark) {
    computeCatmullClarkPositions();
  } else {
    computeLinearSubdivisionPositions();
  }

  // Step II: Assign a unique index (starting at 0) to each vertex, edge,
  // and
  // face in the original mesh.  These indices will be the indices of the
  // vertices
  // in the new (subdivided mesh).  They do not have to be assigned in any
  // particular
  // order, so long as no index is shared by more than one mesh element, and the
  // total number of indices is equal to V+E+F, i.e., the total number of
  // vertices
  // plus edges plus faces in the original mesh.  Basically we just need a
  // one-to-one
  // mapping between original mesh elements and subdivided mesh vertices.
  assignSubdivisionIndices();

  // Step III: Build a list of quads in the new (subdivided) mesh, as
  // tuples of
  // the element indices defined above.  In other words, each new quad should be
  // of
  // the form (i,j,k,l), where i,j,k and l are four of the indices stored on our
  // original mesh elements.  Note that it is essential to get the orientation
  // right
  // here: (i,j,k,l) is not the same as (l,k,j,i).  Indices of new faces should
  // circulate in the same direction as old faces (think about the right-hand
  // rule).
  vector<vector<Index> > subDFaces;
  vector<Vector3D> subDVertices;
  buildSubdivisionFaceList(subDFaces);
  buildSubdivisionVertexList(subDVertices);

  // Step IV: Pass the list of vertices and quads to a routine that clears
  // the
  // internal data for this halfedge mesh, and builds new halfedge data from
  // scratch,
  // using the two lists.
  rebuild(subDFaces, subDVertices);
}

/**
 * Compute new vertex positions for a mesh that splits each polygon
 * into quads (by inserting a vertex at the face midpoint and each
 * of the edge midpoints).  The new vertex positions will be stored
 * in the members Vertex::newPosition, Edge::newPosition, and
 * Face::newPosition.  The values of the positions are based on
 * simple linear interpolation, e.g., the edge midpoints and face
 * centroids.
 */
void HalfedgeMesh::computeLinearSubdivisionPositions() {
    // For each vertex, assign Vertex::newPosition to
    // its original position, Vertex::position.
    for (auto& v : vertices) {
        v.newPosition = v.position;
    }

    // For each edge, assign the midpoint of the two original
    // positions to Edge::newPosition.
    for (auto& e : edges) {
        Vector3D v0 = e.halfedge()->vertex()->position;
        Vector3D v1 = e.halfedge()->twin()->vertex()->position;
        e.newPosition = (v0 + v1) / 2;
    }

    // For each face, assign the centroid (i.e., arithmetic mean)
    // of the original vertex positions to Face::newPosition.  Note
    // that in general, NOT all faces will be triangles!
    for (auto& f : faces) {
        Vector3D center(0);
        HalfedgeIter h = f.halfedge();
        do {
            center += h->vertex()->position;
            h = h->next();
        } while (h != f.halfedge());
        center /= f.degree();
        f.newPosition = center;
    }

    // showError("computeLinearSubdivisionPositions() not implemented.");
}

/**
 * Compute new vertex positions for a mesh that splits each polygon
 * into quads (by inserting a vertex at the face midpoint and each
 * of the edge midpoints).  The new vertex positions will be stored
 * in the members Vertex::newPosition, Edge::newPosition, and
 * Face::newPosition.  The values of the positions are based on
 * the Catmull-Clark rules for subdivision.
 */
void HalfedgeMesh::computeCatmullClarkPositions() {
    // The implementation for this routine should be
    // a lot like HalfedgeMesh::computeLinearSubdivisionPositions(),
    // except that the calculation of the positions themsevles is
    // slightly more involved, using the Catmull-Clark subdivision
    // rules. (These rules are outlined in the Developer Manual.)

    // face
    for (auto& f : faces) {
        Vector3D center(0);
        HalfedgeIter h = f.halfedge();
        do {
            center += h->vertex()->position;
            h = h->next();
        } while (h != f.halfedge());
        center /= f.degree();
        f.newPosition = center;
    }

    // edges
    for (auto& e : edges) {
        HalfedgeIter h0 = e.halfedge();
        HalfedgeIter h1 = h0->twin();
        Vector3D v0 = h0->vertex()->position;
        Vector3D v1 = h1->vertex()->position;
        Vector3D v2, v3;
        if (!h0->isBoundary())
            v2 = h0->face()->newPosition;
        else
            v2 = h1->face()->newPosition;
        if (!h1->isBoundary())
            v3 = h1->face()->newPosition;
        else
            v3 = h0->face()->newPosition;
        e.newPosition = (v0 + v1 + v2 + v3) / 4;
    }

    // vertices
    for (auto& v : vertices) {
        Vector3D F(0), R(0);
        {
            HalfedgeIter h = v.halfedge();
            do {
                if (!h->face()->isBoundary())
                    F += h->face()->newPosition;
                Vector3D v0 = h->vertex()->position;
                Vector3D v1 = h->twin()->vertex()->position;
                R += (v0 + v1) / 2;
                h = h->twin()->next();
            } while(h != v.halfedge());
            F /= v.degree();
            R /= (v.degree() + int(v.isBoundary()));
        }

        int n = v.degree();
        v.newPosition = (F + 2 * R + (n - 3) * v.position) / n;
    }

    // showError("computeCatmullClarkPositions() not implemented.");
}

/**
 * Assign a unique integer index to each vertex, edge, and face in
 * the mesh, starting at 0 and incrementing by 1 for each element.
 * These indices will be used as the vertex indices for a mesh
 * subdivided using Catmull-Clark (or linear) subdivision.
 */
void HalfedgeMesh::assignSubdivisionIndices() {
    // Start a counter at zero; if you like, you can use the
    // "Index" type (defined in halfedgeMesh.h)

    Index curr = 0;
    // Iterate over vertices, assigning values to Vertex::index
    for (auto& v : vertices) {
        v.index = curr++;
    }

    // Iterate over edges, assigning values to Edge::index
    for (auto& e : edges) {
        e.index = curr++;
    }

    // Iterate over faces, assigning values to Face::index
    for (auto& f : faces) {
        f.index = curr++;
    }

    // showError("assignSubdivisionIndices() not implemented.");
}

/**
 * Build a flat list containing all the vertex positions for a
 * Catmull-Clark (or linear) subdivison of this mesh.  The order of
 * vertex positions in this list must be identical to the order
 * of indices assigned to Vertex::newPosition, Edge::newPosition,
 * and Face::newPosition.
 */
void HalfedgeMesh::buildSubdivisionVertexList(vector<Vector3D>& subDVertices) {
    // Resize the vertex list so that it can hold all the vertices.
    subDVertices.resize(vertices.size() + edges.size() + faces.size());

    // Iterate over vertices, assigning Vertex::newPosition to the
    // appropriate location in the new vertex list.
    for (auto& v : vertices) {
        subDVertices[v.index] = v.newPosition;
    }

    // Iterate over edges, assigning Edge::newPosition to the appropriate
    // location in the new vertex list.
    for (auto& e : edges) {
        subDVertices[e.index] = e.newPosition;
    }

    // Iterate over faces, assigning Face::newPosition to the appropriate
    // location in the new vertex list.
    for (auto& f : faces) {
        subDVertices[f.index] = f.newPosition;
    }

    // showError("buildSubdivisionVertexList() not implemented.");
}

/**
 * Build a flat list containing all the quads in a Catmull-Clark
 * (or linear) subdivision of this mesh.  Each quad is specified
 * by a vector of four indices (i,j,k,l), which come from the
 * members Vertex::index, Edge::index, and Face::index.  Note that
 * the ordering of these indices is important because it determines
 * the orientation of the new quads; it is also important to avoid
 * "bowties."  For instance, (l,k,j,i) has the opposite orientation
 * of (i,j,k,l), and if (i,j,k,l) is a proper quad, then (i,k,j,l)
 * will look like a bowtie.
 */
void HalfedgeMesh::buildSubdivisionFaceList(vector<vector<Index> >& subDFaces) {
    // This routine is perhaps the most tricky step in the construction of
    // a subdivision mesh (second, perhaps, to computing the actual Catmull-Clark
    // vertex positions).  Basically what you want to do is iterate over faces,
    // then for each for each face, append N quads to the list (where N is the
    // degree of the face).  For this routine, it may be more convenient to simply
    // append quads to the end of the list (rather than allocating it ahead of
    // time), though YMMV.  You can of course iterate around a face by starting
    // with its first halfedge and following the "next" pointer until you get
    // back to the beginning.  The tricky part is making sure you grab the right
    // indices in the right order---remember that there are indices on vertices,
    // edges, AND faces of the original mesh.  All of these should get used.  Also
    // remember that you must have FOUR indices per face, since you are making a
    // QUAD mesh!

    // iterate over faces
    for (auto& f : faces) {
        if (f.isBoundary())
            continue;

        // loop around face
        vector<EdgeIter> fEdges;
        vector<VertexIter> fVerts;
        HalfedgeIter h = f.halfedge();
        do {
            fEdges.push_back(h->edge());
            fVerts.push_back(h->vertex());
            h = h->next();
        } while (h != f.halfedge());

        int deg = f.degree();
        for (int i = 0; i < deg; i++) {
            // build lists of four indices for each sub-quad
            vector<Index> face = {f.index, fEdges[(i + deg - 1) % deg]->index, fVerts[i]->index, fEdges[i]->index};
            // append each list of four indices to face list
            subDFaces.push_back(face);
        }
    }

    // showError("buildSubdivisionFaceList() not implemented.");
}

FaceIter HalfedgeMesh::bevelVertex(VertexIter v) {
    // This method should replace the vertex v with a face, corresponding to
    // a bevel operation. It should return the new face.  NOTE: This method is
    // responsible for updating the *connectivity* of the mesh only---it does not
    // need to update the vertex positions.  These positions will be updated in
    // HalfedgeMesh::bevelVertexComputeNewPositions (which you also have to
    // implement!)

    if (v->degree() <= 2) {
        showError("Bevel a vertex with degree no-greater than 2");
        return v->halfedge()->face();
    }

    vector<HalfedgeIter> halfedges, inside;
    vector<VertexIter> verts;
    HalfedgeIter th = v->halfedge();
    do {
        th = th->twin();
        halfedges.push_back(th);
        th = th->next();
        halfedges.push_back(th);

        VertexIter nv = newVertex();
        nv->position = v->position;
        verts.push_back(nv);
        inside.push_back(newHalfedge());
    } while (th != v->halfedge());

    FaceIter f = newFace();
    f->halfedge() = inside[0];

    int deg = v->degree();
    for (int i = 0; i < deg; i++) {
        HalfedgeIter nh = newHalfedge();
        EdgeIter ne = newEdge();

        nh->twin() = inside[i];
        inside[i]->twin() = nh;
        nh->edge() = inside[i]->edge() = ne;
        ne->halfedge() = nh;

        nh->vertex() = verts[i];
        verts[i]->halfedge() = nh;

        inside[i]->vertex() = verts[(i + 1) % deg];
        halfedges[i << 1 | 1]->vertex() = verts[(i + 1) % deg];

        halfedges[i << 1]->next() = nh;
        nh->next() = halfedges[i << 1 | 1];
        inside[i]->next() = inside[(i + deg - 1) % deg];

        inside[i]->face() = f;
        nh->face() = halfedges[i << 1]->face();
    }

    deleteVertex(v);

    // showError("bevelVertex() not implemented.");
    return f;
}

FaceIter HalfedgeMesh::bevelEdge(EdgeIter e) {
    // This method should replace the edge e with a face, corresponding to a
    // bevel operation. It should return the new face.  NOTE: This method is
    // responsible for updating the *connectivity* of the mesh only---it does not
    // need to update the vertex positions.  These positions will be updated in
    // HalfedgeMesh::bevelEdgeComputeNewPositions (which you also have to
    // implement!)

    HalfedgeIter h0 = e->halfedge();
    HalfedgeIter h1 = h0->twin();

    if (e->isBoundary()) {
        showError("Bevel a boundary edge");
        return e->halfedge()->face();
    }
    if (h0->vertex()->degree() + h1->vertex()->degree() - 2 <= 2) {
        showError("Bevel an edge with degree no-greater than 2");
        return e->halfedge()->face();
    }

    vector<HalfedgeIter> halfedges, inside;
    vector<VertexIter> verts;

    HalfedgeIter th = h0->next();
    do {
        th = th->twin();
        halfedges.push_back(th);
        th = th->next();
        halfedges.push_back(th);
    } while (th != h1);
    halfedges.pop_back();
    th = h1->next();
    halfedges.push_back(th);
    do {
        th = th->twin();
        halfedges.push_back(th);
        th = th->next();
        halfedges.push_back(th);
    } while (th != h0);
    halfedges.pop_back();
    halfedges.push_back(h0->next());

    int deg = h0->vertex()->degree() + int(h0->vertex()->isBoundary()) + h1->vertex()->degree() + int(h1->vertex()->isBoundary()) - 2;
    for (int i = 0; i < deg; i++) {
        verts.push_back(newVertex());
        inside.push_back(newHalfedge());
    }

    FaceIter f = newFace();
    f->halfedge() = inside[0];
    for (int i = 0; i < deg; i++) {
        HalfedgeIter nh = newHalfedge();
        EdgeIter ne = newEdge();

        nh->twin() = inside[i];
        inside[i]->twin() = nh;
        nh->edge() = inside[i]->edge() = ne;
        ne->halfedge() = nh;

        nh->vertex() = verts[i];
        verts[i]->halfedge() = nh;

        verts[(i + 1) % deg]->position = halfedges[i << 1 | 1]->vertex()->position;
        inside[i]->vertex() = verts[(i + 1) % deg];
        halfedges[i << 1 | 1]->vertex() = verts[(i + 1) % deg];

        halfedges[i << 1]->next() = nh;
        nh->next() = halfedges[i << 1 | 1];
        inside[i]->next() = inside[(i + deg - 1) % deg];

        inside[i]->face() = f;
        nh->face() = halfedges[i << 1]->face();
        halfedges[i << 1]->face()->halfedge() = nh;
    }

    deleteVertex(h0->vertex());
    deleteVertex(h1->vertex());
    deleteEdge(e);
    deleteHalfedge(h0);
    deleteHalfedge(h1);

    // showError("bevelEdge() not implemented.");
    return f;
}

FaceIter HalfedgeMesh::bevelFace(FaceIter f) {
    // This method should replace the face f with an additional, inset face
    // (and ring of faces around it), corresponding to a bevel operation. It
    // should return the new face.  NOTE: This method is responsible for updating
    // the *connectivity* of the mesh only---it does not need to update the vertex
    // positions.  These positions will be updated in
    // HalfedgeMesh::bevelFaceComputeNewPositions (which you also have to
    // implement!)

    if (f->isBoundary()) {
        showError("Bevel a boundary face");
        return f;
    }

    vector<VertexIter> verts;
    vector<HalfedgeIter> outside, halfedges, inside;
    vector<FaceIter> faces;

    HalfedgeIter th = f->halfedge();
    do {
        outside.push_back(th);
        verts.push_back(newVertex());
        th = th->next();
    } while (th != f->halfedge());

    int deg = f->degree();
    for (int i = 0; i < deg; i++) {
        halfedges.push_back(newHalfedge());
        halfedges.push_back(newHalfedge());
        halfedges.push_back(newHalfedge());

        faces.push_back(newFace());
        faces[i]->halfedge() = outside[i];
        outside[i]->face() = faces[i];
        halfedges[i * 3]->face() = faces[i];
        halfedges[i * 3 + 1]->face() = faces[i];
        halfedges[i * 3 + 2]->face() = faces[i];

        inside.push_back(newHalfedge());
    }

    for (int i = 0; i < deg; i++) {
        EdgeIter ne = newEdge();

        outside[i]->next() = halfedges[i * 3];
        halfedges[i * 3]->next() = halfedges[i * 3 + 1];
        halfedges[i * 3 + 1]->next() = halfedges[i * 3 + 2];
        halfedges[i * 3 + 2]->next() = outside[i];
        inside[i]->next() = inside[(i + 1) % deg];

        inside[i]->twin() = halfedges[i * 3 + 1];
        halfedges[i * 3 + 1]->twin() = inside[i];
        inside[i]->edge() = halfedges[i * 3 + 1]->edge() = ne;
        ne->halfedge() = inside[i];
        inside[i]->face() = f;
        f->halfedge() = inside[i];

        halfedges[i * 3]->vertex() = outside[(i + 1) % deg]->vertex();
        halfedges[i * 3 + 1]->vertex() = verts[(i + 1) % deg];
        halfedges[i * 3 + 2]->vertex() = verts[i];
        inside[i]->vertex() = verts[i];
        verts[i]->halfedge() = inside[i];
        verts[i]->position = outside[i]->vertex()->position;

        EdgeIter me = newEdge();
        me->halfedge() = halfedges[i * 3];
        halfedges[i * 3]->twin() = halfedges[((i + 1) * 3 + 2) % (3 * deg)];
        halfedges[((i + 1) * 3 + 2) % (3 * deg)]->twin() = halfedges[i * 3];
        halfedges[i * 3]->edge() = halfedges[((i + 1) * 3 + 2) % (3 * deg)]->edge() = me;
    }

    // showError("bevelFace() not implemented.");
    return f;
}


void HalfedgeMesh::bevelFaceComputeNewPositions(
    vector<Vector3D>& originalVertexPositions,
    vector<HalfedgeIter>& newHalfedges, double normalShift,
    double tangentialInset) {
    // Compute new vertex positions for the vertices of the beveled face.
    //
    // These vertices can be accessed via newHalfedges[i]->vertex()->position for
    // i = 1, ..., newHalfedges.size()-1.
    //
    // The basic strategy here is to loop over the list of outgoing halfedges,
    // and use the preceding and next vertex position from the original mesh
    // (in the originalVertexPositions array) to compute an offset vertex
    // position.
    //
    // Note that there is a 1-to-1 correspondence between halfedges in
    // newHalfedges and vertex positions
    // in orig.  So, you can write loops of the form
    //
    // for( int i = 0; i < newHalfedges.size(); hs++ )
    // {
    //    Vector3D pi = originalVertexPositions[i]; // get the original vertex
    //    position correponding to vertex i
    // }
    //

    Vector3D c0(0), c1(0);
    vector<Vector3D> orig;
    for (auto e : newHalfedges) {
        orig.push_back(e->twin()->vertex()->position);
        c0 += orig.back();
        c1 += e->vertex()->position;
    }
    c0 /= newHalfedges.size();
    c1 /= newHalfedges.size();

    Vector3D v0 = orig[1] - orig[0];
    Vector3D v1 = orig[2] - orig[1];
    Vector3D normal = cross(v1, v0);
    normal.normalize();
    normal *= normalShift;

    vector<Vector3D> delta;
    for (int i = 0; i < newHalfedges.size(); i++) {
        Vector3D dir = c0 - orig[i];
        // dir.normalize();
        dir *= tangentialInset;
        dir += normal;
        if (tangentialInset > 0 && (newHalfedges[i]->vertex()->position - c1).norm2() <= dir.norm2())
            return;
        if (tangentialInset < 0 && (newHalfedges[i]->vertex()->position - orig[i]).norm2() <= dir.norm2())
            return;
        delta.push_back(dir);
    }

    for (int i = 0; i < newHalfedges.size(); i++) {
        newHalfedges[i]->vertex()->position += delta[i];
    }
}

void HalfedgeMesh::bevelVertexComputeNewPositions(
    Vector3D originalVertexPosition, vector<HalfedgeIter>& newHalfedges,
    double tangentialInset) {
    // Compute new vertex positions for the vertices of the beveled vertex.
    //
    // These vertices can be accessed via newHalfedges[i]->vertex()->position for
    // i = 1, ..., hs.size()-1.
    //
    // The basic strategy here is to loop over the list of outgoing halfedges,
    // and use the preceding and next vertex position from the original mesh
    // (in the orig array) to compute an offset vertex position.

    vector<Vector3D> orig;
    for (auto e : newHalfedges) {
        orig.push_back(e->twin()->vertex()->position);
    }

    vector<Vector3D> delta;
    for (int i = 0; i < newHalfedges.size(); i++) {
        Vector3D dir = orig[i] - originalVertexPosition;
        dir.normalize();
        dir *= tangentialInset;
        if (tangentialInset > 0 && (newHalfedges[i]->vertex()->position - orig[i]).norm2() <= dir.norm2())
            return;
        if (tangentialInset < 0 && (newHalfedges[i]->vertex()->position - originalVertexPosition).norm2() <= dir.norm2())
            return;
        delta.push_back(dir);
    }

    for (int i = 0; i < newHalfedges.size(); i++) {
        newHalfedges[i]->vertex()->position += delta[i];
    }
}

void HalfedgeMesh::bevelEdgeComputeNewPositions(
    vector<Vector3D>& originalVertexPositions,
    vector<HalfedgeIter>& newHalfedges, double tangentialInset) {
    // Compute new vertex positions for the vertices of the beveled edge.
    //
    // These vertices can be accessed via newHalfedges[i]->vertex()->position for
    // i = 1, ..., newHalfedges.size()-1.
    //
    // The basic strategy here is to loop over the list of outgoing halfedges,
    // and use the preceding and next vertex position from the original mesh
    // (in the orig array) to compute an offset vertex position.
    //
    // Note that there is a 1-to-1 correspondence between halfedges in
    // newHalfedges and vertex positions
    // in orig.  So, you can write loops of the form
    //
    // for( int i = 0; i < newHalfedges.size(); i++ )
    // {
    //    Vector3D pi = originalVertexPositions[i]; // get the original vertex
    //    position correponding to vertex i
    // }
    //

    vector<Vector3D> orig;
    for (auto e : newHalfedges) {
        orig.push_back(e->twin()->vertex()->position);
    }

    vector<Vector3D> delta;
    for (int i = 0; i < newHalfedges.size(); i++) {
        Vector3D dir = orig[i] - originalVertexPositions[i];
        dir.normalize();
        dir *= tangentialInset;
        if (tangentialInset > 0 && (newHalfedges[i]->vertex()->position - orig[i]).norm2() <= dir.norm2())
            return;
        if (tangentialInset < 0 && (newHalfedges[i]->vertex()->position - originalVertexPositions[i]).norm2() <= dir.norm2())
            return;
        delta.push_back(dir);
    }

    for (int i = 0; i < newHalfedges.size(); i++) {
        newHalfedges[i]->vertex()->position += delta[i];
    }
}

void HalfedgeMesh::splitPolygons(vector<FaceIter>& fcs) {
  for (auto f : fcs) splitPolygon(f);
}

void HalfedgeMesh::splitPolygon(FaceIter f) {
    // TODO: (meshedit)
    // Triangulate a polygonal face
    showError("splitPolygon() not implemented.");
}

EdgeRecord::EdgeRecord(EdgeIter& _edge) : edge(_edge) {
  // TODO: (meshEdit)
  // Compute the combined quadric from the edge endpoints.
  // -> Build the 3x3 linear system whose solution minimizes the quadric error
  //    associated with these two endpoints.
  // -> Use this system to solve for the optimal position, and store it in
  //    EdgeRecord::optimalPoint.
  // -> Also store the cost associated with collapsing this edg in
  //    EdgeRecord::Cost.
}

void MeshResampler::upsample(HalfedgeMesh& mesh)
// This routine should increase the number of triangles in the mesh using Loop
// subdivision.
{
    // Compute new positions for all the vertices in the input mesh, using
    // the Loop subdivision rule, and store them in Vertex::newPosition.
    // -> At this point, we also want to mark each vertex as being a vertex of the
    //    original mesh.
    // -> Next, compute the updated vertex positions associated with edges, and
    //    store it in Edge::newPosition.
    // -> Next, we're going to split every edge in the mesh, in any order.  For
    //    future reference, we're also going to store some information about which
    //    subdivided edges come from splitting an edge in the original mesh, and
    //    which edges are new, by setting the flat Edge::isNew. Note that in this
    //    loop, we only want to iterate over edges of the original mesh.
    //    Otherwise, we'll end up splitting edges that we just split (and the
    //    loop will never end!)
    // -> Now flip any new edge that connects an old and new vertex.
    // -> Finally, copy the new vertex positions into final Vertex::position.

    // Each vertex and edge of the original surface can be associated with a
    // vertex in the new (subdivided) surface.
    // Therefore, our strategy for computing the subdivided vertex locations is to
    // *first* compute the new positions
    // using the connectivity of the original (coarse) mesh; navigating this mesh
    // will be much easier than navigating
    // the new subdivided (fine) mesh, which has more elements to traverse.  We
    // will then assign vertex positions in
    // the new mesh based on the values we computed for the original mesh.

    // Compute updated positions for all the vertices in the original mesh, using
    // the Loop subdivision rule.
    for (auto v = mesh.verticesBegin(); v != mesh.verticesEnd(); v++) {
        v->isNew = false;
        int n = v->degree() + int(v->isBoundary());
        double u = n == 3 ? 3. / 16. : 3. / (8. * n);
        Vector3D pos = (1 - n * u) * v->position;
        HalfedgeIter h = v->halfedge();
        do {
            h = h->twin();
            pos += u * h->vertex()->position;
            h = h->next();
        } while (h != v->halfedge());
        v->newPosition = pos;
    }

    // Next, compute the updated vertex positions associated with edges.
    for (auto e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++) {
        e->isNew = false;
        double a = 3. / 8., b = 1. / 8.;
        HalfedgeIter h0 = e->halfedge();
        HalfedgeIter h1 = h0->twin();
        Vector3D pos = a * (h0->vertex()->position + h1->vertex()->position);
        if (!e->isBoundary()) {
            h0 = h0->next()->next();
            h1 = h1->next()->next();
            pos += b * (h0->vertex()->position + h1->vertex()->position);
        } else if (h0->isBoundary()) {
            h1 = h1->next()->next();
            pos += b * (h1->vertex()->position + h1->vertex()->position);
        } else {
            h0 = h0->next()->next();
            pos += b * (h0->vertex()->position + h0->vertex()->position);
        }
        e->newPosition = pos;
    }

    // Next, we're going to split every edge in the mesh, in any order.  For
    // future
    // reference, we're also going to store some information about which
    // subdivided
    // edges come from splitting an edge in the original mesh, and which edges are
    // new.
    // In this loop, we only want to iterate over edges of the original
    // mesh---otherwise,
    // we'll end up splitting edges that we just split (and the loop will never
    // end!)
    int n = mesh.nEdges();
    auto e = mesh.edgesBegin();
    for (int i = 0; i < n; i++, e++) {
        auto v0 = e->halfedge()->vertex();
        auto v1 = e->halfedge()->twin()->vertex();
        auto nv = mesh.splitEdge(e);
        nv->isNew = true;
        nv->newPosition = e->newPosition;
        HalfedgeIter h = nv->halfedge();
        do {
            h = h->twin();
            if (h->vertex() != v0 && h->vertex() != v1)
                h->edge()->isNew = true;
            h = h->next();
        } while (h != nv->halfedge());
    }

    // Finally, flip any new edge that connects an old and new vertex.
    for (; e != mesh.edgesEnd(); e++) if (e->isNew) {
        auto v0 = e->halfedge()->vertex();
        auto v1 = e->halfedge()->twin()->vertex();
        if (v0->isNew != v1->isNew) mesh.flipEdge(e);
    }

    // Copy the updated vertex positions to the subdivided mesh.
    for (auto v = mesh.verticesBegin(); v != mesh.verticesEnd(); v++) {
        v->position = v->newPosition;
    }

    // showError("upsample() not implemented.");
}

void MeshResampler::downsample(HalfedgeMesh& mesh) {
  // TODO: (meshEdit)
  // Compute initial quadrics for each face by simply writing the plane equation
  // for the face in homogeneous coordinates. These quadrics should be stored
  // in Face::quadric
  // -> Compute an initial quadric for each vertex as the sum of the quadrics
  //    associated with the incident faces, storing it in Vertex::quadric
  // -> Build a priority queue of edges according to their quadric error cost,
  //    i.e., by building an EdgeRecord for each edge and sticking it in the
  //    queue.
  // -> Until we reach the target edge budget, collapse the best edge. Remember
  //    to remove from the queue any edge that touches the collapsing edge
  //    BEFORE it gets collapsed, and add back into the queue any edge touching
  //    the collapsed vertex AFTER it's been collapsed. Also remember to assign
  //    a quadric to the collapsed vertex, and to pop the collapsed edge off the
  //    top of the queue.
  showError("downsample() not implemented.");
}

void MeshResampler::resample(HalfedgeMesh& mesh) {
  // TODO: (meshEdit)
  // Compute the mean edge length.
  // Repeat the four main steps for 5 or 6 iterations
  // -> Split edges much longer than the target length (being careful about
  //    how the loop is written!)
  // -> Collapse edges much shorter than the target length.  Here we need to
  //    be EXTRA careful about advancing the loop, because many edges may have
  //    been destroyed by a collapse (which ones?)
  // -> Now flip each edge if it improves vertex degree
  // -> Finally, apply some tangential smoothing to the vertex positions
  showError("resample() not implemented.");
}

}  // namespace CMU462
