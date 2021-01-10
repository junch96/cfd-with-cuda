#include "solver.h"

#define N 64

namespace matrix
{
    float dotProduct(
        float *a, float *b,
        int _w, int _h
    ) 
    {
        float result = 0.0;
        for (int i = 0; i < _w*_h; i++)
            result += a[i]*b[i];
        return result;
    }
    
    void matrixVectorProduct(
        float *dst, float *b,
        int _w, int _h,
        const float* _aDiag,
        const float* _aPlusX,
        const float* _aPlusY
    ) 
    {
        for (int y = 0, idx = 0; y < _h; y++) {
            for (int x = 0; x < _w; x++, idx++) {
                float t = _aDiag[idx]*b[idx];
                
                if (x > 0)
                    t += _aPlusX[idx -  1]*b[idx -  1];
                if (y > 0)
                    t += _aPlusY[idx - _w]*b[idx - _w];
                if (x < _w - 1)
                    t += _aPlusX[idx]*b[idx +  1];
                if (y < _h - 1)
                    t += _aPlusY[idx]*b[idx + _w];

                dst[idx] = t;
            }
        }
    }
    
    void scaledAdd(
        float *dst, 
        float *a, float *b, float s,
        int _w, int _h
    ) {
        for (int i = 0; i < _w*_h; i++)
            dst[i] = a[i] + b[i]*s;
    }
    
    /* Returns maximum absolute value in vector `a' */
    float infinityNorm(float *a,
        int _w, int _h
    ) {
        float maxA = 0.0;
        for (int i = 0; i < _w*_h; i++)
            maxA = std::max(maxA, float(fabs(a[i])));
        return maxA;
    }

    __global__ void dotProduct_gpu(
        float* dstBlock, 
        float* a,
        float* b
    )
    {
        __shared__ float dst_shared[256];
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        dst_shared[threadIdx.x] = a[x] * b[x];
        __syncthreads();

        for(int len=1;len<blockDim.x;len*=2){
            int j = 2 * len * threadIdx.x;
            if(j < blockDim.x){
                dst_shared[j] += dst_shared[j + len];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0)
            dstBlock[blockIdx.x] = dst_shared[0];
    }

    __global__ void scalarAdd_gpu(
        float* dst, 
        float* src1, 
        float* src2, 
        float c
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        dst[x] = src1[x] + src2[x] * c;
    }

    void __global__ rowMargin(
        int _tx, float* src,
        int w, int h
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * w + x;
        __shared__ float shd[256];
        shd[threadIdx.x] = src[i];
        __syncthreads();
        if(x % _tx == 1 && x != 1)
        {
            shd[threadIdx.x - 2] = shd[threadIdx.x];
        }
        else if(x % _tx == _tx - 2 && x != w - 2)
        {
            shd[threadIdx.x + 2] = shd[threadIdx.x];
        }
        __syncthreads();
        src[i] = shd[threadIdx.x];
    }

    void __global__ colMargin(
        int _ty, float* src,
        int w, int h
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * w + x;
        __shared__ float shd[256];
        shd[threadIdx.y] = src[i];
        __syncthreads();
        if(y % _ty == 1 && y != 1)
        {
            shd[threadIdx.y - 2] = shd[threadIdx.y];
        }
        else if(y % _ty == _ty - 2 && y != h - 2)
        {
            shd[threadIdx.y + 2] = shd[threadIdx.y];
        }
        __syncthreads();
        src[i] = shd[threadIdx.y];
    }

    void __global__ matMul_gpu_tmp(
        int w, int h,
        float* aDig, float* aX, float* aY,
        float* dst, float *src
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * w + x;

        float result = 0.0f;

        if(
            0 < threadIdx.x 
            && threadIdx.x < blockDim.x - 1
            && 0 < threadIdx.y
            && threadIdx.y < blockDim.y - 1
        ){
            result = result + src[i] * aDig[i];
            if(x > 1)
                result = result + src[i-1] * aX[i-1];
            if(y > 1)
                result = result + src[i-w] * aY[i-w];
            if(x < w - 2)
                result = result + src[i+1] * aX[i];
            if(y < h - 2)
                result = result + src[i+w] * aY[i];

            dst[i] = result;
        }
        else{
            dst[i] = 0.0f;
        }
    }

    void __global__ matMul_gpu(
        int w, int h,
        float* aDig, float* aX, float* aY,
        float* dst, float *src
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * w + x;

        __shared__ float aDigShd[16][16];
        __shared__ float aXShd[16][16];
        __shared__ float aYShd[16][16];
        __shared__ float srcShd[16][16];

        aDigShd[threadIdx.y][threadIdx.x] = aDig[i];
        aXShd[threadIdx.y][threadIdx.x] = aX[i];
        aYShd[threadIdx.y][threadIdx.x] = aY[i];
        srcShd[threadIdx.y][threadIdx.x] = src[i];
        
        __syncthreads();

        if(
            0 < threadIdx.x 
            && threadIdx.x < blockDim.x - 1
            && 0 < threadIdx.y
            && threadIdx.y < blockDim.y - 1
        ){
            float result = 0.0f;
            
            result += srcShd[threadIdx.y][threadIdx.x] * aDigShd[threadIdx.y][threadIdx.x];
            
            result += srcShd[threadIdx.y][threadIdx.x - 1] * aXShd[threadIdx.y][threadIdx.x - 1];
            result += srcShd[threadIdx.y - 1][threadIdx.x] * aYShd[threadIdx.y - 1][threadIdx.x];

            result += srcShd[threadIdx.y][threadIdx.x + 1] * aXShd[threadIdx.y][threadIdx.x];
            result += srcShd[threadIdx.y + 1][threadIdx.x] * aYShd[threadIdx.y][threadIdx.x];

            dst[i] = result;
        }
        else{
            dst[i] = 0.0f;
        }
    }

    void __global__ simp_gpu(
        int w, int h,
        float *d,
        float *a
    )
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * w + x;

        __shared__ float as[16][16];
        
        as[threadIdx.x][threadIdx.y] = a[i];
        
        __syncthreads();

        int cx = threadIdx.x;
        int cy = threadIdx.y;

        int lx = threadIdx.x - 1;
        if(threadIdx.x == 0)
            lx = 0;
        int rx = threadIdx.x + 1;
        if(threadIdx.x == blockDim.x - 1)
            rx = threadIdx.x - 1;

        int by = threadIdx.y - 1;
        if(threadIdx.y == 0)
            by = 0;
        int uy = threadIdx.y + 1;
        if(threadIdx.y == blockDim.y - 1)
            uy = threadIdx.y - 1;
        
        float result = 0.0f;

        result += as[cx][cy];
        result += as[cx][by];
        result += as[cx][uy];
        result += as[lx][cy];
        result += as[rx][cy];
        d[i] = result;
    }
}

namespace modified
{
    void buildPreconditioner(
        const int _w,
        const int _h,
        const float* _aPlusX,
        const float* _aPlusY,
        const float* _aDiag,
        float* _precon,
        const float tau = 0.97,
        const float sigma = 0.25
    ) {
        for (int y = 0, idx = 0; y < _h; y++) {
            for (int x = 0; x < _w; x++, idx++) {
                float e = _aDiag[idx];

                if (x > 0) {
                    float px = _aPlusX[idx - 1]*_precon[idx - 1];
                    float py = _aPlusY[idx - 1]*_precon[idx - 1];
                    e = e - (px*px + tau*px*py);
                }
                if (y > 0) {
                    float px = _aPlusX[idx - _w]*_precon[idx - _w];
                    float py = _aPlusY[idx - _w]*_precon[idx - _w];
                    e = e - (py*py + tau*px*py);
                }

                if (e < sigma*_aDiag[idx])
                    e = _aDiag[idx];

                _precon[idx] = 1.0/sqrt(e);
            }
        }
    }
    
    /* Apply preconditioner to vector `a' and store it in `dst' */
    void applyPreconditioner(
        float *dst, float *a,
        const int _w,
        const int _h,
        const float* _aPlusX, 
        const float* _aPlusY,
        const float* _precon
    ) {
        for (int y = 0, idx = 0; y < _h; y++) {
            for (int x = 0; x < _w; x++, idx++) {
                float t = a[idx];

                if (x > 0)
                    t -= _aPlusX[idx -  1]*_precon[idx -  1]*dst[idx -  1];
                if (y > 0)
                    t -= _aPlusY[idx - _w]*_precon[idx - _w]*dst[idx - _w];

                dst[idx] = t*_precon[idx];
            }
        }

        for (int y = _h - 1, idx = _w*_h - 1; y >= 0; y--) {
            for (int x = _w - 1; x >= 0; x--, idx--) {
                idx = x + y*_w;
                
                
                float t = dst[idx];

                if (x < _w - 1)
                    t -= _aPlusX[idx]*_precon[idx]*dst[idx +  1];
                if (y < _h - 1)
                    t -= _aPlusY[idx]*_precon[idx]*dst[idx + _w];

                dst[idx] = t*_precon[idx];
            }
        }
    }
}

namespace incomplete
{
    void applyPreconditioner(
        float *_z, float *_r, 
        const int _w,
        const int _h,
        const float* _aDiag,
        const float* _aPlusX, 
        const float* _aPlusY
    ) 
    {
        for(int x=0;x<_w;x++)
        {
            for(int y=0;y<_h;y++)
            {
                int idx = y * _w + x;
                _z[idx] = _r[idx];
                //_z[idx] = (_r[idx]) / (_aDiag[idx]);
            }
        }
    }
}

float solver::length(float x, float y) 
{
    return sqrt(x*x + y*y);
}

float solver::cubicPulse(float x) 
{
    x = std::min(float(abs(x)), 1.0f);
    return 1.0f - x*x*(3.0f - 2.0f*x);
}
    
float solver::FluidQuantity::lerp(
    float a, 
    float b, 
    float x
) const 
{
    return a*(1.0 - x) + b*x;
}
        
float solver::FluidQuantity::cerp(float a, float b, float c, float d, float x) const {
    float xsq = x*x;
    float xcu = xsq*x;
    
    float minV = std::min({a, b, c, d});
    float maxV = std::max({a, b, c, d});

    float t =
        a*(0.0 - 0.5*x + 1.0*xsq - 0.5*xcu) +
        b*(1.0 + 0.0*x - 2.5*xsq + 1.5*xcu) +
        c*(0.0 + 0.5*x + 2.0*xsq - 1.5*xcu) +
        d*(0.0 + 0.0*x - 0.5*xsq + 0.5*xcu);
    
    return std::min(std::max(t, minV), maxV);
}

void solver::FluidQuantity::rungeKutta3(
    float &x, float &y, float timestep, 
    const FluidQuantity &u, 
    const FluidQuantity &v
) 
const 
{
    float firstU = u.lerp(x, y)/_hx;
    float firstV = v.lerp(x, y)/_hx;

    float midX = x - 0.5*timestep*firstU;
    float midY = y - 0.5*timestep*firstV;

    float midU = u.lerp(midX, midY)/_hx;
    float midV = v.lerp(midX, midY)/_hx;

    float lastX = x - 0.75*timestep*midU;
    float lastY = y - 0.75*timestep*midV;

    float lastU = u.lerp(lastX, lastY);
    float lastV = v.lerp(lastX, lastY);
    
    x -= timestep*((2.0/9.0)*firstU + (3.0/9.0)*midU + (4.0/9.0)*lastU);
    y -= timestep*((2.0/9.0)*firstV + (3.0/9.0)*midV + (4.0/9.0)*lastV);
}
        
solver::FluidQuantity::FluidQuantity(
    int w, int h, 
    float ox, float oy, float hx
)
        : _w(w), _h(h), _ox(ox), _oy(oy), _hx(hx) {
    _src = new float[_w*_h];
    _dst = new float[_w*_h];
            
    memset(_src, 0, _w*_h*sizeof(float));
}

solver::FluidQuantity::~FluidQuantity() 
{
    delete[] _src;
    delete[] _dst;
}

void solver::FluidQuantity::flip() {
    std::swap(_src, _dst);
}

const float* 
    solver::FluidQuantity::src() const 
{
    return _src;
}

float 
    solver::FluidQuantity::at(
        int x, int y
) 
const 
{
    return _src[x + y*_w];
}

float& 
    solver::FluidQuantity::at(
        int x, int y
)
{
    return _src[x + y*_w];
}

float 
    solver::FluidQuantity::lerp(
        float x, float y
) const 
{
    x = std::min(std::max(x - _ox, 0.0f), _w - 1.001f);
    y = std::min(std::max(y - _oy, 0.0f), _h - 1.001f);
    int ix = (int)x;
    int iy = (int)y;
    x -= ix;
    y -= iy;
    
    float x00 = at(ix + 0, iy + 0), x10 = at(ix + 1, iy + 0);
    float x01 = at(ix + 0, iy + 1), x11 = at(ix + 1, iy + 1);
    
    return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y);
}

float 
    solver::FluidQuantity::cerp(
        float x, float y
    ) 
const 
{
    x = std::min(std::max(x - _ox, 0.0f), _w - 1.001f);
    y = std::min(std::max(y - _oy, 0.0f), _h - 1.001f);
    int ix = (int)x;
    int iy = (int)y;
    x -= ix;
    y -= iy;
    
    int x0 = std::max(ix - 1, 0), x1 = ix, x2 = ix + 1, x3 = std::min(ix + 2, _w - 1);
    int y0 = std::max(iy - 1, 0), y1 = iy, y2 = iy + 1, y3 = std::min(iy + 2, _h - 1);
    
    float q0 = cerp(at(x0, y0), at(x1, y0), at(x2, y0), at(x3, y0), x);
    float q1 = cerp(at(x0, y1), at(x1, y1), at(x2, y1), at(x3, y1), x);
    float q2 = cerp(at(x0, y2), at(x1, y2), at(x2, y2), at(x3, y2), x);
    float q3 = cerp(at(x0, y3), at(x1, y3), at(x2, y3), at(x3, y3), x);
    
    return cerp(q0, q1, q2, q3, y);
}

void 
    solver::FluidQuantity::advect(
        float timestep, 
        const FluidQuantity &u, 
        const FluidQuantity &v
) {
    for (int iy = 0, idx = 0; iy < _h; iy++) {
        for (int ix = 0; ix < _w; ix++, idx++) {
            float x = ix + _ox;
            float y = iy + _oy;
            
            rungeKutta3(x, y, timestep, u, v);
            
            _dst[idx] = cerp(x, y);
        }
    }
}

void 
    solver::FluidQuantity::addInflow(
        float x0, float y0, 
        float x1, float y1, 
        float v
) 
{
    int ix0 = (int)(x0/_hx - _ox);
    int iy0 = (int)(y0/_hx - _oy);
    int ix1 = (int)(x1/_hx - _ox);
    int iy1 = (int)(y1/_hx - _oy);
    
    for (int y = std::max(iy0, 0); y < std::min(iy1, _h); y++) {
        for (int x = std::max(ix0, 0); x < std::min(ix1, _h); x++) {
            float l = length(
                (2.0*(x + 0.5)*_hx - (x0 + x1))/(x1 - x0),
                (2.0*(y + 0.5)*_hx - (y0 + y1))/(y1 - y0)
            );
            float vi = cubicPulse(l)*v;
            if (fabs(_src[x + y*_w]) < fabs(vi))
                _src[x + y*_w] = vi;
        }
    }
}
    
int 
solver::FluidSolver::getIdxC(
    int _bx, int _by,
    int _ux, int _uy
)
{
    if(0 <= _bx && _bx < _wQ &&  0 <= _by && _by < _hQ)
    {
        --_ux;
        --_uy;

        int x = _bx * (_tx - 2) + _ux;
        int y = _by * (_ty - 2) + _uy;
    
        return y * _w + x;
    }
    else
    {
        return -1;    
    }
}

int 
solver::FluidSolver::getIdxG(
    int _bx, int _by,
    int _ux, int _uy
)
{
    if(0 <= _bx && _bx < _wQ &&  0 <= _by && _by < _hQ)
    {
        int x = _bx * _tx + _ux;
        int y = _by * _ty + _uy;
        return y * _wExt + x;
    }
    else
    {
        return -1;
    }
}

void 
solver::FluidSolver::makeIdxCtoG()
{
    for(int _bx = 0; _bx < _wQ; ++_bx)
    {
        for(int _by = 0; _by < _hQ; ++_by)
        {
            for(int _ux = 0; _ux < _tx; ++_ux)
            {
                for(int _uy = 0; _uy < _ty; ++_uy)
                {
                    int new_bx = _bx;
                    int new_by = _by;
                    int new_ux = _ux;
                    int new_uy = _uy;

                    int idxG = getIdxG(_bx, _by, _ux, _uy);
                    isMargin[idxG] = false;

                    if(_ux == 0)
                    {
                        new_bx--;
                        new_ux = _tx - 2;
                        isMargin[idxG] = true;
                    }
                    if(_ux == _tx - 1)
                    {
                        new_bx++;
                        new_ux = 1;
                        isMargin[idxG] = true;
                    }
                    if(_uy == 0)
                    {
                        new_by--;
                        new_uy = _ty - 2;    
                        isMargin[idxG] = true;
                    }
                    if(_uy == _ty - 1)
                    {
                        new_by++;
                        new_uy = 1;
                        isMargin[idxG] = true;
                    }
                    
                    int idxC = getIdxC(new_bx, new_by, new_ux, new_uy);
                    
                    idxCtoG[idxG] = idxC;
                }
            }
        }
    }
}

void 
solver::FluidSolver::buildRhs() {
    float scale = 1.0/_hx;
    
    for (int y = 0, idx = 0; y < _h; y++) {
        for (int x = 0; x < _w; x++, idx++) {
            _r[idx] = -scale*(_u->at(x + 1, y) - _u->at(x, y) +
                            _v->at(x, y + 1) - _v->at(x, y));
        }
    }
}

void 
solver::FluidSolver::buildPressureMatrix(
    float timestep
) 
{
    float scale = timestep/(_density*_hx*_hx);
    
    memset(_aDiag, 0, _w*_h*sizeof(float));

    for (int y = 0, idx = 0; y < _h; y++) {
        for (int x = 0; x < _w; x++, idx++) {
            if (x < _w - 1) {
                _aDiag [idx    ] +=  scale;
                _aDiag [idx + 1] +=  scale;
                _aPlusX[idx    ]  = -scale;
            } else
                _aPlusX[idx] = 0.0;

            if (y < _h - 1) {
                _aDiag [idx     ] +=  scale;
                _aDiag [idx + _w] +=  scale;
                _aPlusY[idx     ]  = -scale;
            } else
                _aPlusY[idx] = 0.0;
        }
    }
}

void 
solver::FluidSolver::project_modified(
    int limit
) 
{
    modified::buildPreconditioner(
        _w, _h, _aPlusX, _aPlusY, _aDiag,_precon
    );
    memset(_p, 0,  _w*_h*sizeof(float)); /* Initial guess of zeroes */
    modified::applyPreconditioner(
        _z, _r,
        _w, _h, _aPlusX, _aPlusY, _precon
    );
    memcpy(_s, _z, _w*_h*sizeof(float));
    
    float maxError = matrix::infinityNorm(_r, _w, _h);
    if (maxError < 1e-5)
        return;
    
    float sigma = matrix::dotProduct(_z, _r, _w, _h);
    
    for (int iter = 0; iter < limit; iter++) {
        matrix::matrixVectorProduct(
            _z, _s,
            _w, _h, _aDiag, _aPlusX, _aPlusY
        );
        float alpha = sigma/matrix::dotProduct(_z, _s, _w, _h);
        matrix::scaledAdd(
            _p, _p, _s, alpha, _w, _h
        );
        matrix::scaledAdd(
            _r, _r, _z, -alpha, _w, _h
        );
        
        maxError = matrix::infinityNorm(_r, _w, _h);
        if (maxError < 1e-5) {
            printf("Exiting solver after %d iterations, maximum error is %f\n", iter, maxError);
            return;
        }
        
        modified::applyPreconditioner(
            _z, _r,
            _w, _h, _aPlusX, _aPlusY, _precon
        );
        
        float sigmaNew = matrix::dotProduct(
            _z, _r, _w, _h
        );
        matrix::scaledAdd(
            _s, _z, _s, sigmaNew/sigma, _w, _h
        );
        sigma = sigmaNew;
    }
    
    printf("Exceeded budget of %d iterations, maximum error was %f\n", limit, maxError);
}

void 
solver::FluidSolver::project_incomplete(
    int limit
)
{
    //[1] _r is already initialized

    //[2] 
    memset(_p, 0,  _w*_h*sizeof(float)); /* Initial guess of zeroes */
    
    incomplete::applyPreconditioner(
        _z, _r,
        _w, _h, _aDiag, _aPlusY, _aPlusX
    );
    memcpy(_s, _z, _w*_h*sizeof(float));
    float sigma = matrix::dotProduct(_z, _r, _w, _h);
    
    float maxError = float(-1);

    for (int iter = 0; iter < limit; iter++) {
        matrix::matrixVectorProduct(
            _z, _s,
            _w, _h, _aDiag, _aPlusX, _aPlusY
        );
        float alpha = sigma/matrix::dotProduct(_z, _s, _w, _h);
        
        matrix::scaledAdd(
            _p, _p, _s, alpha, _w, _h
        );
        matrix::scaledAdd(
            _r, _r, _z, -alpha, _w, _h
        );
        maxError = matrix::infinityNorm(_r, _w, _h);
        if (maxError < 1e-5) {
            printf("Exiting solver after %d iterations, maximum error is %f\n", iter, maxError);
            return;
        }
        incomplete::applyPreconditioner(
            _z, _r,
            _w, _h, 
            _aDiag, _aPlusX, _aPlusY
        );
        float sigmaNew = matrix::dotProduct(
            _z, _r, _w, _h
        );
        matrix::scaledAdd(
            _s, _z, _s, sigmaNew/sigma, _w, _h
        );
        sigma = sigmaNew;
    }
}

void 
solver::FluidSolver::moveToExt(
    float* gpuData,
    float* cpuData,
    bool fillMargin
)
{
    for(int idx=0;idx<_whExt;idx++)
    {
        int idxCpu = idxCtoG[idx];
        if(
            isMargin[idx] &&
            !fillMargin
        )
        {
            d_tmp[idx] = 0.0f;
        }
        else if(idxCpu < 0)
        {
            d_tmp[idx] = 0.0f;
        }
        else
        {
            //if(isMatrix)printf("<%d,%d>\n", idx, idxCpu);
            d_tmp[idx] = cpuData[idxCpu];
        }
    }
    cudaMemcpy(gpuData, d_tmp, _whExtSize, cudaMemcpyHostToDevice);
}

void 
solver::FluidSolver::moveFromExt(
    float* cpuData,
    float* gpuData
)   
{
    cudaMemcpy(
        d_tmp,gpuData,_whExtSize,
        cudaMemcpyDeviceToHost
    );
    int cidx = 0;
    for(int y=0;y<_hExt;++y){
        for(int x=0;x<_wExt;++x){
            int gidx = y * _wExt + x;
            int ux = x % _tx;
            int uy = y % _ty;
            if(
                0 < ux && ux < _tx - 1 &&
                0 < uy && uy < _ty - 1
            )
            {
                cpuData[cidx++] = d_tmp[gidx];
            }
        }
    }
}

float 
solver::FluidSolver::dotProduct_parallel(
    float* d_a, float* d_b
)
{
    {
        dim3 threadsPerBlock((_tx*_ty));
        dim3 numBlocks(_whExt / (_tx*_ty));
        matrix::dotProduct_gpu<<<numBlocks, threadsPerBlock>>>(
            val_per_block_device, d_a, d_b
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaMemcpy(
            val_per_block_host,
            val_per_block_device,
            _whExtQSize,cudaMemcpyDeviceToHost
        );
    }
    
    float re = 0.0f;
    for(int idxB=0;idxB<_whExtQ;idxB++){
        re += val_per_block_host[idxB];
    }
    return re;
}

void 
solver::FluidSolver::scalarAdd_parallel(
    float* d_dst,
    float* d_src1,
    float* d_src2,
    float alpha
)
{
    {
        dim3 threadsPerBlock((_tx*_ty));
        dim3 numBlocks(_whExt / (_tx*_ty));
        matrix::scalarAdd_gpu<<<numBlocks, threadsPerBlock>>>(
            d_dst, 
            d_src1, d_src2,
            alpha
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    }
}

void 
solver::FluidSolver::matMul_parallel(
    int w, int h,
    float* d_aDiag, float* d_aX, float* d_aY,
    float* d_dst, 
    float* d_src,
    float* s_src
)
{
    {
        cudaMemcpy(d_e, d_src, _whExtSize, cudaMemcpyDeviceToDevice);
        dim3 threadsPerBlock(_wExt, 1);
        dim3 numBlocks(1, _hExt);
        
        matrix::rowMargin<<<numBlocks, threadsPerBlock>>>(
            _tx, d_e, _wExt, _hExt
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }                
    }
    {
        
        dim3 threadsPerBlock(1, _hExt);
        dim3 numBlocks(_wExt, 1);
        matrix::colMargin<<<numBlocks, threadsPerBlock>>>(
            _ty, d_e, _wExt, _hExt
        );
    }
    {
        dim3 threadsPerBlock(_tx, _ty);
        dim3 numBlocks(_wExt/_tx, _hExt/_ty);
        matrix::matMul_gpu<<<numBlocks, threadsPerBlock>>>(
            _wExt, _hExt,
            d_aDiag, d_aX, d_aY,
            d_dst, d_e
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    }
}

void 
solver::FluidSolver::project_cuda(
    int limit
)
{    
    {//[1]. _r is already initialized.
        moveToExt(d_r,_r, false);
        moveFromExt(
            _r, d_r
        );
    }
    {//[2]. _p
        memset(_p, 0,  _w*_h*sizeof(float)); /* Initial guess of zeroes */
        moveToExt(d_p,_p, false);
    } 
    {//[3]. _z
        moveToExt(d_z,_r, false);
    }
    {//[4]. _s
        moveToExt(d_s,_r, false);
    }
    {//[5]. matrix A
        moveToExt(d_aDiag, _aDiag, true);
        moveToExt(d_aPlusX, _aPlusX, true);
        moveToExt(d_aPlusY, _aPlusY, true);
    }
    {
        float sigma = dotProduct_parallel(d_z, d_r);
        
        for(int iter=0;iter<limit;iter++)
        {

            matMul_parallel(
                _wExt,_hExt,
                d_aDiag,d_aPlusX,d_aPlusY,
                d_z, d_s, _s
            );

            float t = dotProduct_parallel(
                d_z, d_s
            );
            float alpha = sigma/t;
            scalarAdd_parallel(
                d_p, d_p, d_s, alpha
            );
            scalarAdd_parallel(
                d_r, d_r, d_z, -alpha
            );
            cudaMemcpy(
                d_z, d_r, _whExtSize, 
                cudaMemcpyDeviceToDevice
            );
            float sigmaNew = dotProduct_parallel(
                d_z, d_r
            );
            scalarAdd_parallel(
                d_s, d_z, d_s, sigmaNew/sigma
            );
            sigma = sigmaNew;
        }
    }
    {//[6].
        moveFromExt(
            _p, d_p
        );
    }
}

void 
solver::FluidSolver::applyPressure(
    float timestep
) 
{
    float scale = timestep/(_density*_hx);
    
    for (int y = 0, idx = 0; y < _h; y++) {
        for (int x = 0; x < _w; x++, idx++) {
            _u->at(x,     y    ) -= scale*_p[idx];
            _u->at(x + 1, y    ) += scale*_p[idx];
            _v->at(x,     y    ) -= scale*_p[idx];
            _v->at(x,     y + 1) += scale*_p[idx];
        }
    }
    
    for (int y = 0; y < _h; y++)
        _u->at(0, y) = _u->at(_w, y) = 0.0;
    for (int x = 0; x < _w; x++)
        _v->at(x, 0) = _v->at(x, _h) = 0.0;
}

solver::FluidSolver::FluidSolver(
    int w, int h, float density,
    int tx, int ty
) : 
    _w(w), _h(h), _density(density), _tx(tx), _ty(ty) 
{
    //[1]. CPU
    _hx = 1.0/min(w, h);
    
    _d = new FluidQuantity(_w,     _h,     0.5, 0.5, _hx);
    _u = new FluidQuantity(_w + 1, _h,     0.0, 0.5, _hx);
    _v = new FluidQuantity(_w,     _h + 1, 0.5, 0.0, _hx);
    
    _r = new float[_w*_h];
    _p = new float[_w*_h];
    _z = new float[_w*_h];
    _s = new float[_w*_h];
    _aDiag  = new float[_w*_h];
    _aPlusX = new float[_w*_h];
    _aPlusY = new float[_w*_h];
    _precon = new float[_w*_h];
    _aCombined = new float[3*_w*_h];

    //[2]. GPU
    _wQ = _w / (_tx - 2);
    _hQ = _h / (_ty - 2);
    _wExt = _wQ * _tx;
    _hExt = _hQ * _ty;

    _whExt = _wExt * _hExt;
    _whExtSize = _whExt * sizeof(float);

    cudaMalloc((void**)&d_r,_whExtSize);
    cudaMalloc((void**)&d_p,_whExtSize);
    cudaMalloc((void**)&d_z,_whExtSize);
    cudaMalloc((void**)&d_s,_whExtSize);

    cudaMalloc((void**)&d_precon,_whExtSize);
    cudaMalloc((void**)&d_aDiag,_whExtSize);
    cudaMalloc((void**)&d_aPlusX,_whExtSize);
    cudaMalloc((void**)&d_aPlusY,_whExtSize);

    cudaMalloc((void**)&d_e,_whExtSize);

    cudaMalloc((void**)&d_aCombined,4 * _whExtSize);
    
    d_tmp = new float[_whExt];
    d_tmp_combined = new float[4*_whExt];

    //[3]. CPU and GPU
    idxCtoG = new int[_whExtSize];

    isMargin = new bool[_whExt];

    _whExtQ = _wQ * _hQ;    
    _whExtQSize = _wQ * _hQ * sizeof(float);
    val_per_block_host = new float[_whExtQ];
    cudaMalloc((void**)&val_per_block_device,_whExtQSize);

    makeIdxCtoG();
}

solver::FluidSolver::~FluidSolver() 
{
    delete _d;
    delete _u;
    delete _v;
    
    delete[] _r;
    delete[] _p;
    delete[] _z;
    delete[] _s;
    delete[] _aDiag;
    delete[] _aPlusX;
    delete[] _aPlusY;
    delete[] _precon;
}

template < typename T >
void 
solver::FluidSolver::print_cpu_data(
    int w, int h, 
    T* arr
)
{
    int idx = 0;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            std::cout << '\t' << arr[idx++]; 
        }
        puts("");
    }
}
    
void 
solver::FluidSolver::print_gpu_data(
    int w, int h, 
    float* arr
)
{
    cudaMemcpy(
        d_tmp, arr, sizeof(float) * w * h, 
        cudaMemcpyDeviceToHost
    );
    int idx = 0;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            std::cout << '\t' << d_tmp[idx++]; 
        }
        puts("");
    }
}

void 
solver::FluidSolver::update_cpu(
    float timestep,
    bool measureTIme
)
{
    double st, ed;
    if(measureTIme)st = clock();
    {
        buildRhs();
        buildPressureMatrix(timestep);
    }
    if(measureTIme)ed = clock();            
    if(measureTIme)printf("[1]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);

    if(measureTIme)st = clock();
    {
        project_incomplete(100);
    }
    if(measureTIme)ed = clock();            
    if(measureTIme)printf("[2]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);

    if(measureTIme)st = clock();
    {
        applyPressure(timestep);
    }
    if(measureTIme)ed = clock();            
    if(measureTIme)printf("[3]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);
    
    if(measureTIme)st = clock();
    {
        _d->advect(timestep, *_u, *_v);
        _u->advect(timestep, *_u, *_v);
        _v->advect(timestep, *_u, *_v);

        _d->flip();
        _u->flip();
        _v->flip();
    }
    if(measureTIme)ed = clock();        
    if(measureTIme)printf("[4]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);
}

void 
solver::FluidSolver::update_gpu(
    float timestep,
    bool measureTIme
)
{
    double st, ed;
    if(measureTIme)st = clock();
    {
        buildRhs();
        buildPressureMatrix(timestep);
    }
    if(measureTIme)ed = clock();            
    if(measureTIme)printf("[1]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);

    if(measureTIme)st = clock();
    {
        project_cuda(100);
    }
    if(measureTIme)ed = clock();            
    if(measureTIme)printf("[2]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);

    if(measureTIme)st = clock();
    {
        applyPressure(timestep);
    }
    if(measureTIme)ed = clock();            
    if(measureTIme)printf("[3]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);
    

    if(measureTIme)st = clock();
    {
        _d->advect(timestep, *_u, *_v);
        _u->advect(timestep, *_u, *_v);
        _v->advect(timestep, *_u, *_v);

        _d->flip();
        _u->flip();
        _v->flip();
    }
    if(measureTIme)ed = clock();        
    if(measureTIme)printf("[4]time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);
}

void 
solver::FluidSolver::test(
    float timestep
) 
{
    buildRhs();
    buildPressureMatrix(timestep);

    print_cpu_data(
        _wExt, _hExt, idxCtoG
    );

    for(int i=0;i<_w*_h;i++){
        if(i&1)
            _r[i] = (float)(1.0/0.8) * 1;
        else
            _r[i] = (float)(1.0/0.8) * (-1);
    }
    moveToExt(d_r, _r, false);
    print_cpu_data(_w, _h, _r);
    print_gpu_data(_wExt, _hExt, d_r);

    puts("aDiag");
    print_cpu_data(_w, _h, _aDiag);
    moveToExt(d_aDiag, _aDiag, true);
    print_gpu_data(_wExt, _hExt, d_aDiag);
    
    puts("aPlusX");
    print_cpu_data(_w, _h, _aPlusX);
    moveToExt(d_aPlusX, _aPlusX, true);
    print_gpu_data(_wExt, _hExt, d_aPlusX);

    puts("aPlusY");
    print_cpu_data(_w, _h, _aPlusY);
    moveToExt(d_aPlusY, _aPlusY, true);
    print_gpu_data(_wExt, _hExt, d_aPlusY);
    
    matrix::matrixVectorProduct(
        _z, _r,
        _w, _h, _aDiag, _aPlusX, _aPlusY
    );
    printf("Ap from C:%f\n",matrix::dotProduct(_z, _z, _w, _h));

    matMul_parallel(
        _wExt,_hExt,
        d_aDiag,d_aPlusX,d_aPlusY,
        d_s, d_r, _r
    );
    moveFromExt(_s,d_s);
    printf("Ap from G:%f(C)\n",matrix::dotProduct(_s, _s, _w, _h));
    printf("Ap from G:%f(G)\n",
        dotProduct_parallel(d_s, d_s)
    );
    
    puts("z from cpu");
    print_cpu_data(
        _w, _h, _z
    );
    puts("s from gpu");
    print_cpu_data(
        _w, _h, _s
    );

    {//[1] project_mpodified
        puts("project_modified");
        double st, ed;
        st = clock();
        project_incomplete(100);
        ed = clock();            
        printf("time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);
    }
    
    {//[2] project_cuda
        puts("project_cuda");
        double st, ed;
        st = clock();
        //project_cuda(100);
        ed = clock();            
        printf("time : %lf\n",(double)(ed - st)/CLOCKS_PER_SEC);
    }

    applyPressure(timestep);
    
    _d->advect(timestep, *_u, *_v);
    _u->advect(timestep, *_u, *_v);
    _v->advect(timestep, *_u, *_v);
    
    _d->flip();
    _u->flip();
    _v->flip();
}        
    
void 
solver::FluidSolver::addInflow(
    float x, float y, 
    float w, float h, 
    float d, 
    float u, float v
)
{
    _d->addInflow(x, y, x + w, y + h, d);
    _u->addInflow(x, y, x + w, y + h, u);
    _v->addInflow(x, y, x + w, y + h, v);
}
    
void 
solver::FluidSolver::dump(
    float *vertices
)
{
    int idx = 0;
    int idxG = 0;
    for(int iy=0;iy<_h;iy++){
        for(int ix=0;ix<_w;ix++){
            float len_y = 2.0/_h;
            float len_y_half = 1.0/_h;
            float len_x = 2.0/_w;
            float len_x_half = 1.0/_w;

            float x = -1.0f + len_x_half + len_x * ix;
            float y = -1.0f + len_y_half + len_y * iy;
            float z = 0.0f;

            int shade = (int)((1.0 - _d->src()[idxG++])*255.0);
            shade = max(min(shade, 255), 0);
            float d = float(shade)/255;
            
            //printf("_srd->d : %f\n", _d->src()[idxG - 1]);
            //printf("d : %f\n", d);

            vertices[idx++] = x;
            vertices[idx++] = y;
            vertices[idx++] = z;
            vertices[idx++] = d;
            vertices[idx++] = d;
            vertices[idx++] = d;
        }
    }
}

void 
solver::FluidSolver::toImage(
    unsigned char *rgba
) 
{
    for (int i = 0; i < _w*_h; i++) {
        int shade = (int)((1.0 - _d->src()[i])*255.0);
        shade = max(min(shade, 255), 0);
        
        rgba[i*4 + 0] = shade;
        rgba[i*4 + 1] = shade;
        rgba[i*4 + 2] = shade;
        rgba[i*4 + 3] = 0xFF;
    }
}  
