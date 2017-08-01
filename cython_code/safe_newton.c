#include <math.h>
#include "safe_newton.h"
#define MAXIT 100 /*Maximum allowed number of iterations.*/

float rtsafe(void (*funcd)(float, float *, float *), float x1, float x2, float xacc)
/*Using a combination of Newton-Raphson and bisection, find the root of a function bracketed
between x1 and x2. The root, returned as the function value rtsafe, will be refined until
its accuracy is known within ±xacc. funcd is a user-supplied routine that returns both the
function value and the first derivative of the function.*/
{
    void nrerror(char error_text[]);
    int j;
    float df,dx,dxold,f,fh,fl;
    float temp,xh,xl,rts;

    (*funcd)(x1,&fl,&df);
    (*funcd)(x2,&fh,&df);
    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0))
        nrerror("Root must be bracketed in rtsafe");
    if (fl == 0.0) return x1;
    if (fh == 0.0) return x2;
    if (fl < 0.0) { /*Orient the search so that f(xl) < 0.*/
        xl=x1;
        xh=x2;
    } else {
        xh=x1;
        xl=x2;
    }
    rts=0.5*(x1+x2); /*Initialize the guess for root,*/
    dxold=fabs(x2-x1); /*the “stepsize before last,”*/
    dx=dxold; /*and the last step.*/
    (*funcd)(rts,&f,&df);
    for (j=1;j<=MAXIT;j++) { /*Loop over allowed iterations.*/
        if ((((rts-xh)*df-f)*((rts-xl)*df-f) > 0.0) /*Bisect if Newton out of range,*/
            || (fabs(2.0*f) > fabs(dxold*df))) { /*or not decreasing fast enough.*/
            dxold=dx;
            dx=0.5*(xh-xl);
            rts=xl+dx;
            if (xl == rts) return rts; /*Change in root is negligible.*/
        } else { /*Newton step acceptable. Take it.*/
            dxold=dx;
            dx=f/df;
            temp=rts;
            rts -= dx;
            if (temp == rts) return rts;
        }
        if (fabs(dx) < xacc) return rts; /*Convergence criterion.*/
        (*funcd)(rts,&f,&df);
        /*The one new function evaluation per iteration.*/
        if (f < 0.0) /*Maintain the bracket on the root.*/
            xl=rts;
        else
            xh=rts;
    }
    nrerror("Maximum number of iterations exceeded in rtsafe");
    return 0.0; /*Never get here.*/
}