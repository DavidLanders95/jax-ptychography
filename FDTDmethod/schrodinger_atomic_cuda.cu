
/*********************************************************************************/
/*                                                                               */
/*  Relaxation to Ground State of Atomic Potential (Imaginary Time)              */
/*  CUDA ACCELERATED VERSION                                                     */
/*                                                                               */
/*********************************************************************************/

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* General geometrical parameters */
#define WINWIDTH 	1280  
#define WINHEIGHT 	720   
#define NX 640          
#define NY 360          

#define XMIN -2.0f
#define XMAX 2.0f	
#define YMIN -1.125f
#define YMAX 1.125f

/* Color parameters */
#define COLOR_SCHEME 3

/* Physical parameters */
#define DT 0.000001f
#define DX ((XMAX-XMIN)/((float)NX))

/* Relativistic / Helmholtz Parameters */
#define SOL_LIGHT 137.0f      /* Speed of light (approx au) */
#define MASS_PART 1.0f        /* Particle Mass */
#define E_KINETIC 300.0f      /* Beam Energy */
/* Total Energy E = mc^2 + T */
#define E_TOTAL (MASS_PART*SOL_LIGHT*SOL_LIGHT + E_KINETIC)

/* Derived Beam Parameters */
/* p^2 c^2 = E^2 - m^2 c^4 */
/* p = sqrt(E^2 - m^2c^4)/c */
#define P_INF (sqrtf(E_TOTAL*E_TOTAL - MASS_PART*MASS_PART*SOL_LIGHT*SOL_LIGHT*SOL_LIGHT*SOL_LIGHT) / SOL_LIGHT)
/* In our non-rel simulator units (hbar=1, m=1), omega = k^2/2. */
/* We simulate the equation: i dpsi/dt = -0.5 lap psi + V_eff psi 
   To match Helmholtz (lap + k^2)psi = 0, we look for steady state psi(t) = phi(x) e^-iEt
   E phi = -0.5 lap phi + V_eff phi  => lap phi + 2(E - V_eff) phi = 0
   We want lap phi + k_local^2 phi = 0.
   So k_local^2 = 2(E - V_eff) => V_eff = E - 0.5 k_local^2.
   We choose Simulation Energy E_SIM = 0.5 * P_INF * P_INF (non-relativistic equivalent energy for the beam k)
*/
#define MOMENTUM_X P_INF
#define E_SIM (0.5f * MOMENTUM_X * MOMENTUM_X)

/* Initial Wave Packet (Now Continuous Source) */
#define X_START -1.8f
#define Y_START 0.0f
#define ATOMIC_Z 10.0f  /* Very strong potential to see relativistic effects */
#define SOFTENING 0.25f    

/* CUDA Error Checking Macro */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Device Helper: Coordinate Mapping */
__device__ void ij_to_xy_cuda(int i, int j, float *x, float *y)
{
    *x = XMIN + ((float)i)*(XMAX-XMIN)/((float)NX);
    *y = YMIN + ((float)j)*(YMAX-YMIN)/((float)NY);
}

/* Device Helper: Atomic Potential */
/* Modified to read from global memory array */
__device__ float potential_cuda(float *d_V, int i, int j)
{
    return d_V[j * NX + i];
}

/* KERNEL: Update Phi using Psi */
__global__ void step1_update_phi(float *d_phi, float *d_psi, float *d_V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        int idx = j * NX + i; // Standard row-major layout
        
        /* Periodic Boundary Conditions */
        int ip = (i + 1) % NX;
        int im = (i - 1 + NX) % NX;
        int jp = (j + 1) % NY;
        int jm = (j - 1 + NY) % NY;

        // Laplacian stencil
        float y_curr = d_psi[idx];
        float delta2 = d_psi[j*NX + ip] + d_psi[j*NX + im] + 
                       d_psi[jp*NX + i] + d_psi[jm*NX + i] - 4.0f * y_curr;
        
        float V = potential_cuda(d_V, i, j);
        
        float intstep = DT/(DX*DX);
        
        /* phi[i][j] += ( -0.5 * intstep * delta2 + V * y * DT ); */
        d_phi[idx] += ( -0.5f * intstep * delta2 + V * y_curr * DT );
        
        /* Calculate Position for Sponge */
        float x_pos, y_pos;
        ij_to_xy_cuda(i, j, &x_pos, &y_pos);

        /* Absorbing Boundary Layer (Sponge) at RIGHT limit ONLY */
        /* We rely on periodic wrapping into the right sponge to absorb left-traveling waves */
        float sponge_width = 0.5f;
        float dist_right = fmaxf(x_pos - (XMAX - sponge_width), 0.0f);
        float sponge_dist = dist_right;
        
        if (sponge_dist > 0.0f) {
            /* Normalized distance 0..1 in the sponge */
            float r = sponge_dist / sponge_width;
            /* Cubic ramp for smoothness */
            float profile = r * r * r; 
            /* High velocity waves require stronger damping. 
               v ~ 24 au. L ~ 0.5. T ~ 0.02. 
               Need exp(-V*T) ~ 1e-6 => V*T ~ 14 => V ~ 700.
               Let's set peak V_imag to 5000.0 to be safe. */
            float damping = 5000.0f * profile * DT;
            
            d_phi[idx] *= (1.0f - damping);
        }
    }
}

/* KERNEL: Source Injection */
__global__ void inject_source(float *d_phi, float *d_psi, float time)
{
    /* Inject Plane Wave at Left Boundary */
    /* Sponge width is 0.5 (indices 0-80). 
       Place source at i=0. Left-traveling waves wrap to right sponge. */
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_source = 0; 
    
    if (j < NY) {
        int idx = j * NX + i_source;
        /* Inject Plane Wave: exp(-i E t) */
        /* Smooth turn-on to avoid shock? No, continuous wave is fine. */
        d_phi[idx] = cosf(-E_SIM * time); 
        d_psi[idx] = sinf(-E_SIM * time);
    }
}
__global__ void step2_update_psi(float *d_phi, float *d_psi, float *d_V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        int idx = j * NX + i;
        
        /* Periodic Boundary Conditions */
        int ip = (i + 1) % NX;
        int im = (i - 1 + NX) % NX;
        int jp = (j + 1) % NY;
        int jm = (j - 1 + NY) % NY;
        
        // Laplacian stencil on NEW phi
        float x_curr = d_phi[idx]; 
        float delta1 = d_phi[j*NX + ip] + d_phi[j*NX + im] + 
                       d_phi[jp*NX + i] + d_phi[jm*NX + i] - 4.0f * x_curr;
        
        float V = potential_cuda(d_V, i, j);
         
        float intstep = DT/(DX*DX);

        /* psi[i][j] += ( 0.5 * intstep * delta1 - V * x * DT ); */
        d_psi[idx] += ( 0.5f * intstep * delta1 - V * x_curr * DT );
        
        /* Calculate Position for Sponge */
        float x_pos, y_pos;
        ij_to_xy_cuda(i, j, &x_pos, &y_pos);

        /* Absorbing Boundary Layer (Sponge) at RIGHT limit ONLY */
        float sponge_width = 0.5f;
        float dist_right = fmaxf(x_pos - (XMAX - sponge_width), 0.0f);
        float sponge_dist = dist_right;
        
        if (sponge_dist > 0.0f) {
            float r = sponge_dist / sponge_width;
            float profile = r * r * r; 
            float damping = 5000.0f * profile * DT;
            d_psi[idx] *= (1.0f - damping);
        }
    }
}


/* Host Functions for IO (reused/adapted from headless C) */
void write_ppm(char *filename, int width, int height, unsigned char *data) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error opening %s for writing\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height * 3, fp);
    fclose(fp);
    printf("Saved %s\n", filename);
}

// Minimal HSL to RGB helper for host side
void hsl_to_rgb(float h, float s, float l, float *r, float *g, float *b) {
    float c = (1.0f - fabsf(2.0f*l - 1.0f)) * s;
    float x = c * (1.0f - fabsf(fmodf(h/60.0f, 2.0f) - 1.0f));
    float m = l - c/2.0f;
    
    if(0<=h && h<60){ *r=c; *g=x; *b=0; }
    else if(60<=h && h<120){ *r=x; *g=c; *b=0; }
    else if(120<=h && h<180){ *r=0; *g=c; *b=x; }
    else if(180<=h && h<240){ *r=0; *g=x; *b=c; }
    else if(240<=h && h<300){ *r=x; *g=0; *b=c; }
    else{ *r=c; *g=0; *b=x; }
    
    *r += m; *g += m; *b += m;
}

void color_scheme(float phi, float psi, float scale, int time, float *rgb) {
    // Scheme 3: Modulus -> Lightness, Phase -> Hue
    float mod = sqrtf(phi*phi + psi*psi);
    float val = mod * scale;
    if (val > 1.0f) val = 1.0f;
    
    float angle = atan2f(psi, phi) * 180.0f / 3.14159f;
    if(angle < 0) angle += 360.0f;
    
    hsl_to_rgb(angle, 1.0f, 0.5f * val, &rgb[0], &rgb[1], &rgb[2]);
}

int main(int argc, char **argv)
{
    int size = NX * NY * sizeof(float);
    float *h_phi = (float*)malloc(size);
    float *h_psi = (float*)malloc(size);
    /* Load Potential Map */
    float *h_V = (float*)malloc(size);
    
    FILE *fp_pot = fopen("potential.bin", "rb");
    if(!fp_pot) {
        printf("Error: potential.bin not found. Please provide input potential map.\n");
        return 1;
    }
    // Skip dimensions check for now, assume NX x NY
    fseek(fp_pot, 8, SEEK_SET); 
    size_t read_count = fread(h_V, sizeof(float), NX*NY, fp_pot);
    if(read_count != NX*NY) {
         printf("Warning: Potential file size mismatch? Read %zu floats.\n", read_count);
    }
    fclose(fp_pot);

    int i, frame;

    /* Parse command-line overrides: N_FRAMES STEPS_PER_FRAME SNAPSHOT_COUNT */
    int N_FRAMES = 200;
    int STEPS_PER_FRAME = 50;
    int SNAPSHOT_COUNT = 10;
    if (argc > 1) N_FRAMES = atoi(argv[1]);
    if (argc > 2) STEPS_PER_FRAME = atoi(argv[2]);
    if (argc > 3) SNAPSHOT_COUNT = atoi(argv[3]);

    printf("Initializing CUDA Atomic Simulation (User Potential)...\n");
    printf("Run: N_FRAMES=%d, STEPS_PER_FRAME=%d, SNAPSHOT_COUNT=%d\n", N_FRAMES, STEPS_PER_FRAME, SNAPSHOT_COUNT);

    /* Initial Condition (Host) - Start with Vacuum */
    for (i=0; i<NX*NY; i++) {
        h_phi[i] = 0.0f;
        h_psi[i] = 0.0f;
    }
    
    /* Device Memory */
    float *d_phi, *d_psi, *d_V;
    cudaCheckError( cudaMalloc((void**)&d_phi, size) );
    cudaCheckError( cudaMalloc((void**)&d_psi, size) );
    cudaCheckError( cudaMalloc((void**)&d_V, size) );
    
    /* Copy to Device */
    cudaCheckError( cudaMemcpy(d_phi, h_phi, size, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_psi, h_psi, size, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice) );
    
    /* Config */
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /* Prepare snapshots file (will store last SNAPSHOT_COUNT frames) */
    FILE *fp_snap = fopen("snapshots.bin", "wb");
    if (!fp_snap) {
        printf("Error: cannot open snapshots.bin for writing\n");
        return 1;
    }
    int nx = NX, ny = NY, M = SNAPSHOT_COUNT;
    float dt_f = DT;
    float E_sim_f = E_SIM;
    fwrite(&nx, sizeof(int), 1, fp_snap);
    fwrite(&ny, sizeof(int), 1, fp_snap);
    fwrite(&M, sizeof(int), 1, fp_snap);
    fwrite(&dt_f, sizeof(float), 1, fp_snap);
    fwrite(&E_sim_f, sizeof(float), 1, fp_snap);

    /* Determine start frame to save snapshots (last M frames) */
    int save_start = N_FRAMES - M;
    if (save_start < 0) save_start = 0;
    int saved = 0;

    /* Main Time Loop */
    for (frame=0; frame < N_FRAMES; frame++) {
        for (int s=0; s<STEPS_PER_FRAME; s++) {
            float time = (frame * STEPS_PER_FRAME + s) * DT;
            
            inject_source<<<numBlocks, threadsPerBlock>>>(d_phi, d_psi, time);
            
            step1_update_phi<<<numBlocks, threadsPerBlock>>>(d_phi, d_psi, d_V);
            cudaCheckError( cudaPeekAtLastError() );
            
            step2_update_psi<<<numBlocks, threadsPerBlock>>>(d_phi, d_psi, d_V);
            cudaCheckError( cudaPeekAtLastError() );
        }
        if (frame % 20 == 0) printf("Progress: %d / %d frames\n", frame, N_FRAMES);

        /* Save snapshot if in the last M frames (at frame granularity) */
        if (frame >= save_start && saved < M) {
            /* copy device arrays back */
            cudaCheckError( cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost) );
            cudaCheckError( cudaMemcpy(h_psi, d_psi, size, cudaMemcpyDeviceToHost) );
            float t_snapshot = (frame * STEPS_PER_FRAME) * DT;
            fwrite(&t_snapshot, sizeof(float), 1, fp_snap);
            fwrite(h_phi, sizeof(float), NX*NY, fp_snap);
            fwrite(h_psi, sizeof(float), NX*NY, fp_snap);
            saved++;
        }
    }

    fclose(fp_snap);
    printf("Saved %d snapshots to snapshots.bin\n", saved);

    /* Also save final summed wavefunction for compatibility */
    cudaCheckError( cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost) );
    cudaCheckError( cudaMemcpy(h_psi, d_psi, size, cudaMemcpyDeviceToHost) );

    FILE *fp_bin = fopen("wavefunction.bin", "wb");
    if (fp_bin) {
        fwrite(&nx, sizeof(int), 1, fp_bin);
        fwrite(&ny, sizeof(int), 1, fp_bin);
        fwrite(h_phi, sizeof(float), NX*NY, fp_bin);
        fwrite(h_psi, sizeof(float), NX*NY, fp_bin);
        fclose(fp_bin);
        printf("Saved final wavefunction to wavefunction.bin\n");
    }

    cudaFree(d_phi);
    cudaFree(d_psi);
    cudaFree(d_V);
    free(h_phi);
    free(h_psi);
    free(h_V);
    
    return 0;
}
