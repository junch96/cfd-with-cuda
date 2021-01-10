#ifndef SOLVER
#define SOLVER
#include <algorithm>
#include <cstdio>
#include <iostream>
namespace solver{
    float length(float x, float y);
    float cubicPulse(float x) ;

    class FluidQuantity {
    private:
        float *_src;
        float *_dst;

        int _w;
        int _h;
        float _ox;
        float _oy;
        float _hx;
        
        float lerp(float a, float b, float x) const;
        
        float cerp(float a, float b, float c, float d, float x) const;
        
        void rungeKutta3(
            float &x, float &y, float timestep, 
            const FluidQuantity &u, 
            const FluidQuantity &v
        ) 
        const;
    public:
        FluidQuantity(int w, int h, float ox, float oy, float hx);
        ~FluidQuantity();
        void flip();
        
        const float *src() const;
        
        float at(int x, int y) const;
        
        float &at(int x, int y);
        
        float lerp(float x, float y) const;
        
        float cerp(float x, float y) const;
        
        void advect(float timestep, 
            const FluidQuantity &u, 
            const FluidQuantity &v
        );
        
        void addInflow(float x0, float y0, float x1, float y1, float v);
    };

    class FluidSolver {
    private:
        FluidQuantity *_d;
        FluidQuantity *_u;
        FluidQuantity *_v;

        int _w;
        int _h;
        float _hx;
        float _density;
        float *_r;
        float *_p;
        float *_z; /* Auxiliary vector */
        float *_s; /* Search vector */
        float *_precon; /* Preconditioner */
        float *_aDiag;  /* Matrix diagonal */
        float *_aPlusX; /* Matrix off-diagonals */
        float *_aPlusY;

        float *_aCombined;

        //[2]. GPU
        int _tx, _ty;
        int _wQ, _hQ;
        int _wExt, _hExt;
        int _whExt;
        int _whExtSize;

        float* d_r;
        float* d_p;
        float* d_z; /* Auxiliary vector */
        float* d_s; /* Search vector */
        float* d_precon; /* Preconditioner */
        float* d_aDiag;  /* Matrix diagonal */
        float* d_aPlusX; /* Matrix off-diagonals */
        float* d_aPlusY;

        float* d_e;

        float* d_aCombined;

        //[3]. CPU and GPU connection;
        float* d_tmp;
        float* d_tmp_combined;

        //[3]. CPU and GPU
        int* idxCtoG;

        bool* isMargin;

        int _whExtQ;
        int _whExtQSize;
        float* val_per_block_host;
        float* val_per_block_device;

        int getIdxC(
            int _bx, int _by,
            int _ux, int _uy
        );

        int getIdxG(
            int _bx, int _by,
            int _ux, int _uy
        );

        void makeIdxCtoG();
        
        void buildRhs();
        
        void buildPressureMatrix(float timestep);
        
        /* Conjugate gradients solver */
        void project_modified(int limit);

        void project_incomplete(int limit);
        
        void moveToExt(
            float* gpuData,
            float* cpuData,
            bool fillMargin = false
        );

        void moveFromExt(
            float* cpuData,
            float* gpuData
        );
        float dotProduct_parallel(
            float* d_a, float* d_b
        );

        void scalarAdd_parallel(
            float* d_dst,
            float* d_src1,
            float* d_src2,
            float alpha
        );
        
        void matMul_parallel(
            int w, int h,
            float* d_aDiag, float* d_aX, float* d_aY,
            float* d_dst, 
            float* d_src,
            float* s_src
        );

        void project_cuda(int limit);

        void applyPressure(float timestep);
    public:
        FluidSolver(
            int w, int h, float density,
            int _tx = 16, int _ty = 16
        );
        ~FluidSolver();

        template < typename T >
        void print_cpu_data(
            int w, int h, 
            T* arr
        );
        
        void print_gpu_data(
            int w, int h, 
            float* arr
        );

        void update_cpu(float timestep, bool measureTIme);

        void update_gpu(float timestep, bool measureTIme);
        
        void test(float timestep);
        
        void addInflow(
            float x, float y, 
            float w, float h, 
            float d, float u, float v
        );
        
        void dump(float *vertices);

        void toImage(unsigned char *rgba);
    };

}

#endif