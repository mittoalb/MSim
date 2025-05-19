// brain_ext.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <cstdint>

namespace py = pybind11;

// Label constants
static constexpr uint8_t CELL_VAL    = 5;
static constexpr uint8_t NUCLEUS_VAL = 7;
static constexpr uint8_t AXON_VAL    = 8;
static constexpr uint8_t VESSEL_VAL  = 5;

//----------------------------------------------------------------------------------------------------------------
// carve_ball_cpu: fill a sphere at (z0,y0,x0)
inline void carve_ball_cpu(
    uint8_t* mask, int nz, int ny, int nx,
    int z0, int y0, int x0, int radius, uint8_t label
) {
    int rr = radius * radius;
    int zmin = std::max(0, z0 - radius), zmax = std::min(nz - 1, z0 + radius);
    int ymin = std::max(0, y0 - radius), ymax = std::min(ny - 1, y0 + radius);
    int xmin = std::max(0, x0 - radius), xmax = std::min(nx - 1, x0 + radius);
    for (int zz = zmin; zz <= zmax; ++zz) {
        int dz = zz - z0, dz2 = dz * dz;
        for (int yy = ymin; yy <= ymax; ++yy) {
            int dy = yy - y0, dy2 = dy * dy;
            for (int xx = xmin; xx <= xmax; ++xx) {
                int dx = xx - x0;
                if (dz2 + dy2 + dx*dx <= rr) {
                    mask[(zz * ny + yy) * nx + xx] = label;
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------
// Gaussian blur 1D helper (for macroregions warp)
static std::vector<double> gaussian_kernel(int radius, double sigma) {
    int size = 2*radius+1;
    std::vector<double> kernel(size);
    double sum = 0.0;
    for(int i=0;i<size;++i) {
        double x = i - radius;
        kernel[i] = std::exp(-(x*x)/(2*sigma*sigma));
        sum += kernel[i];
    }
    for(auto &v:kernel) v /= sum;
    return kernel;
}

//----------------------------------------------------------------------------------------------------------------
// add_macroregions_cpp: warp and assign N horizontal layers
void add_macroregions_cpp(
    py::array_t<uint8_t, py::array::c_style> labels,
    int macro_regions,
    double region_smoothness
) {
    auto buf = labels.request();
    int nz = buf.shape[0], ny = buf.shape[1], nx = buf.shape[2];
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    // Generate simple per-slice noise and blur in X
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0,1.0);
    int radius = int(region_smoothness * 3);
    auto kernel = gaussian_kernel(radius, region_smoothness);
    std::vector<std::vector<double>> warp2d(nz, std::vector<double>(nx));
    // random noise
    for(int z=0;z<nz;++z) for(int x=0;x<nx;++x)
        warp2d[z][x] = dist(gen);
    // blur each row
    std::vector<double> temp(nx);
    for(int z=0;z<nz;++z) {
        for(int x=0;x<nx;++x) {
            double v=0;
            for(int k=-radius;k<=radius;++k) {
                int xx = std::clamp(x+k,0,nx-1);
                v += warp2d[z][xx]*kernel[k+radius];
            }
            temp[x] = v;
        }
        warp2d[z] = temp;
    }
    // assign layers
    double layer_thick = double(ny)/macro_regions;
    for(int z=0;z<nz;++z) {
        for(int y=0;y<ny;++y) {
            for(int x=0;x<nx;++x) {
                double warp = (warp2d[z][x] - 0.5) * (layer_thick*0.4);
                for(int r=0;r<macro_regions;++r) {
                    double lo = r*layer_thick + warp;
                    double hi = (r+1)*layer_thick + warp;
                    if(y>=lo && y<hi) {
                        ptr[(z*ny+y)*nx + x] = uint8_t(r+1);
                        break;
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------
// draw_vessels_cpu: iterative branching with carve_ball
void draw_vessels_cpu(
    uint8_t* mask, int nz, int ny, int nx,
    int root_z, int root_y, int root_x,
    int max_depth, int base_radius,
    const float* rng_vals, int rng_len
) {
    struct Node { int z,y,x,depth,idx; double dx,dy,dz; int radius; };
    std::vector<Node> stack;
    stack.reserve(64);
    stack.push_back({root_z,root_y,root_x,0,0,0.0,0.0,1.0,base_radius});
    while(!stack.empty()) {
        auto n = stack.back(); stack.pop_back();
        if(n.depth>=max_depth || n.idx+3>rng_len) continue;
        // normalize
        double norm = std::sqrt(n.dx*n.dx + n.dy*n.dy + n.dz*n.dz)+1e-6;
        double dx1 = n.dx/norm, dy1=n.dy/norm, dz1=n.dz/norm;
        // compute travel limit
        double t_max=1e9;
        double pos[3]={double(n.z),double(n.y),double(n.x)};
        double dir[3]={dz1,dy1,dx1};
        for(int i=0;i<3;++i) {
            double d=dir[i]; double p=pos[i]; int lim=(i==0?nz:i==1?ny:nx);
            if(d>0) t_max = std::min(t_max,((lim-1)-p)/d);
            else if(d<0) t_max = std::min(t_max,-p/d);
        }
        int length = std::min(int(t_max),80);
        double pz=pos[0], py=pos[1], px=pos[2];
        int zi,yi,xi;
        for(int i=0;i<length;++i) {
            if(i%5==0) {
                dx1 += (rng_vals[n.idx]-0.5)*1.0;
                dy1 += (rng_vals[n.idx+1]-0.5)*1.5;
                dz1 += (rng_vals[n.idx+2]-0.5)*1.0;
                n.idx+=3;
                double nnstd = std::sqrt(dx1*dx1+dy1*dy1+dz1*dz1)+1e-6;
                dx1/=nnstd; dy1/=nnstd; dz1/=nnstd;
            }
            pz+=dz1; py+=dy1; px+=dx1;
            zi=int(pz); yi=int(py); xi=int(px);
            if(zi>=0&&zi<nz&&yi>=0&&yi<ny&&xi>=0&&xi<nx)
                carve_ball_cpu(mask,nz,ny,nx,zi,yi,xi,n.radius,VESSEL_VAL);
        }
        // Fibonacci branching
        if(n.radius>1) {
            int fib0=1,fib1=1;
            for(int i=0;i<n.depth+1;++i) { int t=fib1; fib1+=fib0; fib0=t;} 
            int nb=std::min(2+(fib1%6),5);
            for(int b=0;b<nb;++b){
                int base=n.idx+b*3; if(base+3>rng_len) break;
                double ndx=dx1+(rng_vals[base]-0.5)*2.0;
                double ndy=dy1+(rng_vals[base+1]-0.5)*2.0;
                double ndz=dz1+(rng_vals[base+2]-0.5)*2.0;
                int cr=std::max(1,int(n.radius*(0.5+0.4*rng_vals[base+2])));
                stack.push_back({zi,yi,xi,n.depth+1,base,ndx,ndy,ndz,cr});
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------
// add_vessels_cpp: choose random roots and draw vessels
void add_vessels_cpp(
    py::array_t<uint8_t, py::array::c_style> labels,
    int num_vessels,
    int max_depth,
    int vessel_radius,
    int seed
) {
    auto buf = labels.request();
    int nz=buf.shape[0], ny=buf.shape[1], nx=buf.shape[2];
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist_face(0,5);
    std::uniform_int_distribution<int> dist_z(0,nz-1);
    std::uniform_int_distribution<int> dist_y(0,ny-1);
    std::uniform_int_distribution<int> dist_x(0,nx-1);
    int rng_len=50000;
    std::unique_ptr<float[]> rng_vals(new float[rng_len]);
    std::uniform_real_distribution<float> dist_f(0.0f,1.0f);
    for(int i=0;i<rng_len;++i) rng_vals[i]=dist_f(gen);
    for(int i=0;i<num_vessels;++i) {
        int face = dist_face(gen);
        int rz,ry,rx;
        switch(face){
            case 0: rz=0;    ry=dist_y(gen); rx=dist_x(gen); break;
            case 1: rz=nz-1; ry=dist_y(gen); rx=dist_x(gen); break;
            case 2: rz=dist_z(gen); ry=0;    rx=dist_x(gen); break;
            case 3: rz=dist_z(gen); ry=ny-1; rx=dist_x(gen); break;
            case 4: rz=dist_z(gen); ry=dist_y(gen); rx=0;    break;
            default: rz=dist_z(gen); ry=dist_y(gen); rx=nx-1; break;
        }
        draw_vessels_cpu(ptr,nz,ny,nx,rz,ry,rx,max_depth,vessel_radius,rng_vals.get(),rng_len);
    }
}

//----------------------------------------------------------------------------------------------------------------
PYBIND11_MODULE(brainext, m) {
    m.def("add_macroregions", &add_macroregions_cpp,
          "Add warped macro-regions in C++");
    m.def("add_neurons",    &add_neurons_cpp,
          "Place cells, nuclei, and axons in C++");
    m.def("add_vessels",    &add_vessels_cpp,
          "Generate vascular trees in C++");
}


