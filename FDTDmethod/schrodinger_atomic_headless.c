/*********************************************************************************/
/*                                                                               */
/*  Relaxation to Ground State of Atomic Potential (Imaginary Time)              */
/*  HEADLESS VERSION (No GLUT/OpenGL) - Writes PPM images                        */
/*                                                                               */
/*********************************************************************************/

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

/* General geometrical parameters */
#define WINWIDTH 	640  
#define WINHEIGHT 	360   
#define NX 320          
#define NY 180          

#define XMIN -2.0
#define XMAX 2.0	
#define YMIN -1.125
#define YMAX 1.125	

/* Constants required by colors_waves.c */
#define COLORHUE 260     /* initial hue of water color for scheme C_LUM */
#define COLORDRIFT 0.0   /* how much the color hue drifts during the whole simulation */
#define LUMMEAN 0.5      /* amplitude of luminosity variation for scheme C_LUM */
#define LUMAMP 0.3       /* amplitude of luminosity variation for scheme C_LUM */
#define HUEMEAN 180.0    /* mean value of hue for color scheme C_HUE */
#define HUEAMP 180.0      /* amplitude of variation of hue for color scheme C_HUE */

#define NSTEPS 2500      
#define ATTENUATION 0.0
#define COLOR_RANGE 1.0
#define SLOPE 1.0

/* Physical parameters */
#define DT 0.00001
#define DX ((XMAX-XMIN)/((double)NX))

/* Initial Wave Packet */
#define X_START -1.5
#define Y_START 0.0
#define PACKET_WIDTH_X 0.5   /* Longer pulse */
#define PACKET_WIDTH_Y 6.0   /* Very wide in Y to approximate plane wave */
#define MOMENTUM_X 25.0
#define MOMENTUM_Y 0.0
#define ATOMIC_Z 100.0     /* Much stronger potential for visible scattering */
#define SOFTENING 0.2    /* Wider core */

#define PLOT 0 
/* 0 = Real Part/Amplitude */

#define COLOR_SCHEME 3   
#define COLOR_PALETTE 10

/* Includes from the code base */
#include "global_pdes.c" 
/* We include colors_waves.c but need to ensure we don't pick up GL deps from it? */
/* colors_waves.c contains color_scheme() which calls hsl_to_rgb() etc. Pure math. */
#include "colors_waves.c"

double intstep;

/* Helpers normally in sub_wave.c */
void ij_to_xy(int i, int j, double xy[2])
{
    xy[0] = XMIN + ((double)i)*(XMAX-XMIN)/((double)NX);
    xy[1] = YMIN + ((double)j)*(YMAX-YMIN)/((double)NY);
}

double module2(double x, double y)
{
    return sqrt(x*x + y*y);
}

/* Image Writer */
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

void schrodinger_color_scheme(double phi, double psi, double scale, int time, double rgb[3])
{
    /* Using P_REAL behavior for simplicity or P_MODULE */
    if (PLOT == 0) /* Assuming 0 means Amplitude/Real */
        color_scheme(COLOR_SCHEME, phi, scale, time, rgb);
    else
        color_scheme(COLOR_SCHEME, 2.0*module2(phi, psi)-1.0, scale, time, rgb);
}

void save_image(double *phi[NX], double *psi[NX], short int *xy_in[NX], double scale, int time, int frame_num)
{
    int i, j;
    double rgb[3];
    unsigned char *image = (unsigned char *)malloc(NX * NY * 3);
    char filename[100];
    
    #pragma omp parallel for private(i,j,rgb)
    for (j=0; j<NY; j++) {
        for (i=0; i<NX; i++) {
            // PPM stores top-to-bottom usually, but mathematical j=0 is bottom.
            // Let's store j=0 at bottom (height-1-j) to match standard Cartesian view.
            int pixel_index = ((NY - 1 - j) * NX + i) * 3;
            
            if (xy_in[i][j]) {
                schrodinger_color_scheme(phi[i][j], psi[i][j], scale, time, rgb);
            } else {
                rgb[0] = 0.0; rgb[1] = 0.0; rgb[2] = 0.0;
            }
            
            // Clamp 0..1 to 0..255
            image[pixel_index] = (unsigned char)(rgb[0] * 255.0);
            image[pixel_index+1] = (unsigned char)(rgb[1] * 255.0);
            image[pixel_index+2] = (unsigned char)(rgb[2] * 255.0);
        }
    }
    
    sprintf(filename, "wave_%05d.ppm", frame_num);
    write_ppm(filename, NX, NY, image);
    free(image);
}

void evolve_wave_half_rtp(double *phi_in[NX], double *psi_in[NX], double *phi_out[NX], double *psi_out[NX], 
                      short int *xy_in[NX])
/* Real Time Propagation Step with Potential */
{
	/* Placeholder to keep compiler happy, though we operate inline in main now */
}

void evolve_wave(double *phi[NX], double *psi[NX], short int *xy_in[NX])
{
    /* In-place update using temporary buffers or just ping-pong? 
       Actually standard schrodinger.c uses ping-pong. 
       Let's alloc tmp buffers in main and pass them? 
       For simplicity here, I'll allocate tmp locally (slower) or just modify evolve signature? 
       Let's stick to simple "allocate once in main". 
       Wait, I need to match the signature or just do whatever since libraries are gone.
    */
    /* I'll implement a simple one-step function that takes buffers */
}

int main(int argc, char** argv)
{
    int i, j, frame;
    double *phi[NX], *psi[NX], *phi_tmp[NX], *psi_tmp[NX];
    short int *xy_in[NX];
    
    printf("Initializing Headless Atomic Simulation...\n");
    
    intstep = DT/(DX*DX);
    
    /* Allocation */
    for (i=0; i<NX; i++){
        phi[i] = (double *) malloc(NY*sizeof(double));
        psi[i] = (double *) malloc(NY*sizeof(double));
        phi_tmp[i] = (double *) malloc(NY*sizeof(double));
        psi_tmp[i] = (double *) malloc(NY*sizeof(double));
        xy_in[i] = (short int *) malloc(NY*sizeof(short int));
    }
    
    /* Initialize Grid */
    for (i=0; i<NX; i++)
        for (j=0; j<NY; j++)
            xy_in[i][j] = 1; 

    /* Initial Condition: Traveling Wave Packet */
    printf("Setting Initial Condition: Wave Packet (k_x=%.1f)...\n", MOMENTUM_X);
    for (i=0; i<NX; i++)
        for (j=0; j<NY; j++) {
            double p[2];
            ij_to_xy(i, j, p);
            
            // Gaussian Envelope (Plane Wave in Y -> Infinite extent in Y, limited by box)
            double dx = p[0] - X_START;
            // double dy = p[1] - Y_START; // Removed for plane wave
            double env = exp( -(dx*dx)/(2.0 * PACKET_WIDTH_X * PACKET_WIDTH_X) );
            
            // Ensure strictly zero at boundaries to prevent pinned-boundary reflection artifacts
            double dist_to_bdy = fmin( fmin(p[0]-XMIN, XMAX-p[0]), fmin(p[1]-YMIN, YMAX-p[1]) );
            if (dist_to_bdy < 0.2) {
                 env *= (dist_to_bdy / 0.2) * (dist_to_bdy / 0.2); // Smooth cutoff
            }
            if (dist_to_bdy <= 0.0) env = 0.0;
            
            // Plane Wave Phase
            double phase = MOMENTUM_X * p[0] + MOMENTUM_Y * p[1];
            
            phi[i][j] = env * cos(phase);
            psi[i][j] = env * sin(phase);
        }

    int N_FRAMES = 200;
    int STEPS_PER_FRAME = 50;
    
    for (frame=0; frame<N_FRAMES; frame++)
    {
        /* Auto-scale color */
        double max_val = 0.0;
        /* Check max amplitude in center area or global? Global is safer early on. */
        for (int x=0; x<NX; x++) for(int y=0; y<NY; y++) {
            double amp = module2(phi[x][y], psi[x][y]);
            if(amp > max_val) max_val = amp;
        }
        if (max_val < 1e-10) max_val = 1.0;
        
        save_image(phi, psi, xy_in, 1.0/max_val, frame, frame);
        
        for (int s=0; s<STEPS_PER_FRAME; s++) {
            /* Step 1: Forward Euler / Symplectic Attempt 
               We call evolve_wave_half_rtp which does one explicit step.
               A better scheme for Schrödinger is Split-Operator or Crank-Nicolson, 
               but explicit FD is what schrodinger.c used (effectively 1st order in time).
               Wait, schrodinger.c did:
                   evolve_half(phi_in, psi_in -> phi_out, psi_out)
                   evolve_half(phi_out, psi_out -> phi_in, psi_in) 
               (but actually swapping buffers).
               
               Also note schrodinger.c defined evolve_wave_half to update phi_out from phi_in AND psi_in.
               Actually schrodinger.c did:
               phi_out = phi_in - dt * H * psi_in
               psi_out = psi_in + dt * H * phi_in
               
               This is NOT symplectic if done simultaneously. It's Forward Euler. Unstable for oscillatory?
               However, schrodinger.c works? 
               Wait, schrodinger.c uses: 
               phi_out = x - intstep*delta2 (where delta2 comes from psi_in)
               psi_out = y + intstep*delta1 (where delta1 comes from phi_in)
               Yes, that is Forward Euler. It is conditionally stable for diffusion, but for Schrödinger (imaginary diffusion) it is UNCONDITIONALLY UNSTABLE. 
               
               UNLESS... schrodinger.c uses a different trick or my memory of numerical stability is simplified.
               Actually, explicit Euler is unstable for pure advection/wave equation.
               Maybe schrodinger.c relies on very small DT or the fact that it calls it twice? 
               
               Let's look at standard "Leapfrog" for Schrödinger:
               phi(t+1) = phi(t-1) + 2*dt*H*psi(t)
               
               Or maybe schrodinger.c used:
               phi(t+1) = phi(t) - dt*H*psi(t)
               psi(t+1) = psi(t) + dt*H*phi(t+1)  <-- Semi-implicit / Symplectic Euler
               
               Let's check my implementation of evolve_wave_half_rtp. 
               I am reading x (phi_in) and y (psi_in) and computing phi_out and psi_out simultaneously.
               That is standard Forward Euler.
               
               If I want Symplectic Euler (stable-ish):
               Update phi first using psi_in.
               Then update psi using the NEW phi.
               
               Let's try to implement that.
            */
            
            double delta1, delta2, x, y, r2, V;
            double p[2];

            /* Substep 1: Update Phi using Psi */
            #pragma omp parallel for private(i,j,delta2,y,V,p,r2)
            for (i=1; i<NX-1; i++){
                for (j=1; j<NY-1; j++){
                    if (xy_in[i][j]){
                        y = psi[i][j];
                        delta2 = psi[i+1][j] + psi[i-1][j] + psi[i][j+1] + psi[i][j-1] - 4.0*y;
                        
                        ij_to_xy(i, j, p);
                        r2 = p[0]*p[0] + p[1]*p[1];
                        V = -ATOMIC_Z / sqrt(r2 + SOFTENING*SOFTENING);
                        
                        /* phi_new = phi_old + H psi_old * dt */
                        /* d/dt phi = -0.5 Lap psi + V psi */
                        phi[i][j] += ( -0.5 * intstep * delta2 + V * y * DT );
                        
                        /* Absorbing Boundary Layer (Sponge) - Improved profile */
                        double dist = fmin( fmin(p[0]-XMIN, XMAX-p[0]), fmin(p[1]-YMIN, YMAX-p[1]) );
                        if (dist < 0.4) {
                            double strength = pow((0.4-dist)/0.4, 3); /* Smooth ramping */
                            phi[i][j] *= (1.0 - 0.2 * strength);
                        }
                    }
                }
            }
            
            /* Substep 2: Update Psi using NEW Phi */
            #pragma omp parallel for private(i,j,delta1,x,V,p,r2)
            for (i=1; i<NX-1; i++){
                for (j=1; j<NY-1; j++){
                    if (xy_in[i][j]){
                        x = phi[i][j]; /* This is the UPDATED phi */
                        delta1 = phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1] - 4.0*x;
                        
                        ij_to_xy(i, j, p);
                        r2 = p[0]*p[0] + p[1]*p[1];
                        V = -ATOMIC_Z / sqrt(r2 + SOFTENING*SOFTENING);
                        
                        /* d/dt psi = 0.5 Lap phi - V phi */
                        psi[i][j] += ( 0.5 * intstep * delta1 - V * x * DT );
                        
                        /* Absorbing Boundary Layer (Sponge) */
                        double dist = fmin( fmin(p[0]-XMIN, XMAX-p[0]), fmin(p[1]-YMIN, YMAX-p[1]) );
                        if (dist < 0.4) {
                            double strength = pow((0.4-dist)/0.4, 3);
                            psi[i][j] *= (1.0 - 0.2 * strength);
                        }
                    }
                }
            }
            
            /* No renormalization in RTP! Energy should be conserved naturally approx. */
        }
        
        printf("Frame %d/%d completed. Max amp: %g\n", frame+1, N_FRAMES, max_val);
    }
    
    return 0;
}
