#include "geometry.h"



#include "geometry.h"
#include <cmath>
#include <algorithm>

// Helper: compute two orthonormal vectors u, v perpendicular to dir
static void make_orthonormal_basis(const double dir[3], double u[3], double v[3]) {
    // pick an arbitrary vector not parallel to dir
    double a[3] = { (std::fabs(dir[0])<0.9 ? 1.0 : 0.0),
                    (std::fabs(dir[1])<0.9 ? 1.0 : 0.0),
                    (std::fabs(dir[2])<0.9 ? 1.0 : 0.0) };
    // u = dir × a
    u[0] = dir[1]*a[2] - dir[2]*a[1];
    u[1] = dir[2]*a[0] - dir[0]*a[2];
    u[2] = dir[0]*a[1] - dir[1]*a[0];
    double nu = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) + 1e-6;
    u[0]/=nu; u[1]/=nu; u[2]/=nu;
    // v = dir × u
    v[0] = dir[1]*u[2] - dir[2]*u[1];
    v[1] = dir[2]*u[0] - dir[0]*u[2];
    v[2] = dir[0]*u[1] - dir[1]*u[0];
    double nv = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 1e-6;
    v[0]/=nv; v[1]/=nv; v[2]/=nv;
}

// Original full‐ellipsoid carving (assuming integer radius for minor axes)
void carve_ellipsoid(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int zc, int yc, int xc,
    double length,
    double radius,
    uint8_t label,
    double& total_length
) {
    int Rz = int(std::round(length));
    int Rr = int(std::round(radius));
    // you can reuse your carve_ball/carve_cylinder code here as a
    // quick approximation: carve a sphere of radius Rr, then extrude
    // along z-axis for length Rz.  For brevity, here’s a simple sphere:
    for (int dz = -Rz; dz <= Rz; ++dz) {
        for (int dy = -Rr; dy <= Rr; ++dy) {
            for (int dx = -Rr; dx <= Rr; ++dx) {
                if (dx*dx + dy*dy + dz*dz <= Rr*Rr) {
                    int z = zc + dz, y = yc + dy, x = xc + dx;
                    if (z>=0 && z<nz && y>=0 && y<ny && x>=0 && x<nx) {
                        size_t idx = size_t(z)*ny*nx + size_t(y)*nx + x;
                        labels[idx] = label;
                        occ[idx]    = 1;
                    }
                }
            }
        }
    }
    total_length += 0;  // no length accounting here
}


// Compute (dz,dy,dx) = gradient at voxel idx via 6‐nbr central differences
static void compute_local_normal(
    const uint8_t* occ, int nz, int ny, int nx,
    int z, int y, int x,
    double &dz, double &dy, double &dx
) {
    auto idx = [&](int zz,int yy,int xx){ return size_t(zz)*ny*nx + size_t(yy)*nx + xx; };
    // clamp neighbors
    int zp = std::min(z+1, nz-1), zm = std::max(z-1, 0);
    int yp = std::min(y+1, ny-1), ym = std::max(y-1, 0);
    int xp = std::min(x+1, nx-1), xm = std::max(x-1, 0);
    // central difference on occupancy mask
    dz = double( occ[idx(zp,y,x)] ) - double( occ[idx(zm,y,x)] );
    dy = double( occ[idx(z,y+1,x)] ) - double( occ[idx(z,y-1,x)] );
    dx = double( occ[idx(z,y,xp)] ) - double( occ[idx(z,y,xm)] );
    // flip so it points *into* lumen
    double norm = std::sqrt(dz*dz + dy*dy + dx*dx) + 1e-6;
    dz /= norm; dy /= norm; dx /= norm;
}


#include <cmath>

// ---------------------------------------------------------------------------
// Build an orthonormal basis {n, v, w} given n=(dz,dy,dx)
void make_basis(
    double dz, double dy, double dx,
    double &vx, double &vy, double &vz,
    double &wx, double &wy, double &wz
) {
    // pick a vector not colinear with n
    if (std::fabs(dz) < std::fabs(dx)) {
        vx =  0;    vy = -dx; vz =  dy;
    } else {
        vx = -dy; vy =  dz;  vz =  0;
    }
    // normalize v
    double vn = std::sqrt(vx*vx + vy*vy + vz*vz) + 1e-6;
    vx/=vn; vy/=vn; vz/=vn;

    // w = n × v
    wx = dy*vz - dz*vy;
    wy = dz*vx - dx*vz;
    wz = dx*vy - dy*vx;
    double wn = std::sqrt(wx*wx + wy*wy + wz*wz) + 1e-6;
    wx/=wn; wy/=wn; wz/=wn;
}

void carve_oriented_semi_cylinder(
    uint8_t* labels, uint8_t* occ,
    int nz,int ny,int nx,
    double cz,double cy,double cx,
    double dz,double dy,double dx,
    double half_len,
    double radius,
    uint8_t label,
    double& total_length
) {
    // Build local basis (n=vessel normal, v & w tangent axes)
    double vx,vy,vz, wx,wy,wz;
    make_basis(dz,dy,dx, vx,vy,vz, wx,wy,wz);

    // bounding box around our semi-cylinder
    int R = int(ceil(radius));
    int L = int(ceil(half_len));
    int z0 = int(floor(cz)), y0 = int(floor(cy)), x0 = int(floor(cx));

    for (int dzp = -L; dzp <= L; ++dzp) {
      for (int v = -R; v <= R; ++v) {
       for (int w = -R; w <= R; ++w) {
         // only keep points where v² + w² <= radius² (circle)
         double d2 = double(v)*v + double(w)*w;
         if (d2 > radius*radius) continue;

         // and only the half where v·n_norm > 0 (flat face on wall side)
         // flat face is where (v*dot(n,v)) <= 0 → so keep dot(n,v)>0 side
         if (v*0 + w*0 < 0) {} // dummy; we'll test instead below

         // compute world coordinates: P = C + dzp*n + v*v_axis + w*w_axis
         double X = cx + dzp*dx + v*vx + w*wx;
         double Y = cy + dzp*dy + v*vy + w*wy;
         double Z = cz + dzp*dz + v*vz + w*vz; // careful with axes

         int zi = int(round(Z)), yi=int(round(Y)), xi=int(round(X));
         if (zi<0||zi>=nz||yi<0||yi>=ny||xi<0||xi>=nx) continue;

         // now decide which side of flat face this is on:
         // dot( (P-C) , (v_axis) ) >= 0 → keep rounded side
         double dotv = (v*vx + w*wx);
         if (dotv < 0) continue;

         size_t idx = zi*ny*nx + yi*nx + xi;
         labels[idx] = label;
         occ[idx]    = 1;
         total_length += 1.0;
       }
      }
    }
}



// ---------------------------------------------------------------------------
// Carve one narrow semi‐cylinder wedge around the vessel wall
void carve_semi_cylinder_wedge(
    uint8_t* labels, uint8_t* occ,
    int nz,int ny,int nx,
    double cz,double cy,double cx,
    double dz,double dy,double dx,
    double ux,double uy,double uz,
    double half_len,
    double radius,
    uint8_t label,
    double& total_length
) {
    // precompute perp‐axis w = n × u
    double wx = dy*uz - dz*uy;
    double wy = dz*ux - dx*uz;
    double wz = dx*uy - dy*ux;

    int L = int(std::ceil(half_len));
    int R = int(std::ceil(radius));

    for (int i = -L; i <= L; ++i) {
      for (int a = -R; a <= R; ++a) {
        for (int b = -R; b <= R; ++b) {
          // circle test
          if (double(a)*a + double(b)*b > radius*radius) continue;
          // curved‐side test
          double dotuw = a*ux + b*wx
                        + a*uy + b*wy
                        + a*uz + b*wz;
          if (dotuw < 0) continue;

          // world coords
          double X = cx + i*dx + a*ux + b*wx;
          double Y = cy + i*dy + a*uy + b*wy;
          double Z = cz + i*dz + a*uz + b*wz;

          int zi = int(std::round(Z)),
              yi = int(std::round(Y)),
              xi = int(std::round(X));
          if (zi<0||zi>=nz||yi<0||yi>=ny||xi<0||xi>=nx) continue;

          size_t idx = size_t(zi)*ny*nx + size_t(yi)*nx + xi;
          labels[idx] = label;
          occ[idx]    = 1;
          total_length += 1.0;
        }
      }
    }
}



// ----------------------------------------------------------------------------
// Carve a prolate ellipsoid of “half-length” half_len along the
// axis n=(dz,dy,dx), of radius=r, labeling voxels & marking occ[].
// This signature must match exactly the header declaration:
void carve_oriented_ellipsoid(
    uint8_t* labels, uint8_t* occ,
    int nz,int ny,int nx,
    double cz,double cy,double cx,
    double dz,double dy,double dx,
    double half_len,     // semi-axis along n
    double radius,       // semi-axis perpendicular to n
    uint8_t label,
    double& total_length
) {
    // normalize direction
    double L = std::sqrt(dz*dz + dy*dy + dx*dx) + 1e-6;
    dz /= L; dy /= L; dx /= L;

    // build an orthonormal basis {n, v, w}
    double vx,vy,vz, wx,wy,wz;
    make_basis(dz,dy,dx, vx,vy,vz, wx,wy,wz);

    // bounding box
    int R = int(std::ceil(std::max(half_len, radius)));
    for (int iz = -R; iz <= R; ++iz) {
        for (int i = -R; i <= R; ++i) {
            for (int j = -R; j <= R; ++j) {
                // coordinates in ellipsoid param‐space:
                //   p = cz + iz*n + i*v + j*w
                double X = cx + iz*dx + i*vx + j*wx;
                double Y = cy + iz*dy + i*vy + j*wy;
                double Z = cz + iz*dz + i*vz + j*wz;

                // check ellipsoid equation: (iz/half_len)^2 + (i/radius)^2 + (j/radius)^2 <= 1
                double e = (iz*iz)/(half_len*half_len)
                         + (i*i)/(radius*radius)
                         + (j*j)/(radius*radius);
                if (e > 1.0) continue;

                int zi = int(std::round(Z)),
                    yi = int(std::round(Y)),
                    xi = int(std::round(X));
                if (zi<0||zi>=nz||yi<0||yi>=ny||xi<0||xi>=nx) continue;

                size_t idx = size_t(zi)*ny*nx + size_t(yi)*nx + xi;
                labels[idx] = label;
                occ[idx]    = 1;
                total_length += 1.0;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// carve a full cylinder of radius 'radius' and half‐length 'half_len'
// along the unit‐vector (dz,dy,dx), centered at (cz,cy,cx).
void carve_oriented_cylinder(
    uint8_t* labels, uint8_t* occ,
    int nz, int ny, int nx,
    double cz, double cy, double cx,
    double dz, double dy, double dx,
    double half_len,
    double radius,
    uint8_t label,
    double &total_length
) {
    // build perpendicular basis {v, w} to the axis n=(dz,dy,dx)
    double vx, vy, vz, wx, wy, wz;
    make_basis(dz, dy, dx, vx, vy, vz, wx, wy, wz);

    int L = int(std::ceil(half_len));
    int R = int(std::ceil(radius));
    double r2 = radius * radius;

    for (int i = -L; i <= L; ++i) {
        for (int a = -R; a <= R; ++a) {
            for (int b = -R; b <= R; ++b) {
                // radial test
                if (double(a)*a + double(b)*b > r2) 
                    continue;

                // world coordinates of this voxel
                double X = cx + i*dz + a*vx + b*wx;
                double Y = cy + i*dy + a*vy + b*wy;
                double Z = cz + i*dx + a*vz + b*wz;

                int zi = int(std::round(Z));
                int yi = int(std::round(Y));
                int xi = int(std::round(X));
                if (zi<0 || zi>=nz || yi<0 || yi>=ny || xi<0 || xi>=nx)
                    continue;

                size_t idx = size_t(zi)*ny*nx + size_t(yi)*nx + xi;
                labels[idx] = label;
                occ   [idx] = 1;
                total_length += 1.0;
            }
        }
    }
}


void carve_hollow_ellipsoid(
    uint8_t* labels,
    uint8_t* occ,
    int nz, int ny, int nx,
    int cz, int cy, int cx,
    double rz, double ry, double rx,  // direction vector (ignored if isotropic)
    double outer_radius,
    double thickness,
    uint8_t label,
    double& carved_len
) {
    double inner_radius = std::max(0.0, outer_radius - thickness);
    int r = int(std::ceil(outer_radius));
    for (int dz = -r; dz <= r; ++dz) {
        int z = cz + dz;
        if (z < 0 || z >= nz) continue;
        for (int dy = -r; dy <= r; ++dy) {
            int y = cy + dy;
            if (y < 0 || y >= ny) continue;
            for (int dx = -r; dx <= r; ++dx) {
                int x = cx + dx;
                if (x < 0 || x >= nx) continue;

                double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d >= inner_radius && d <= outer_radius) {
                    size_t idx = (z * ny + y) * nx + x;
                    if (occ[idx]) continue;
                    labels[idx] = label;
                    occ[idx]    = 1;
                }
            }
        }
    }

    carved_len += 1.0;  // Optional: just count one per placement
}
