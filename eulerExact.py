import math
import numpy as np
import matplotlib.pyplot as plt

def riemann_f(p, rho, v, P, gamma, A, B, cs):
    if p <= P:
        f = 2*cs*(math.pow(p/P,(gamma-1)/(2*gamma))-1.0) / (gamma-1.0)
        df = 2*cs*math.pow(p/P,-(gamma+1)/(2*gamma)) / (2*gamma*P)
    else:
        f = (p-P)*math.sqrt(A / (p+B))
        df = (1.0 - 0.5*(p-P)/(p+B)) * math.sqrt(A / (p+B))
    return f, df

def riemann(a, b, x0, N, T, rhoL, vL, PL, rhoR, vR, PR, gamma, 
                TOL=1.0e-14, MAX=100):

    AL = 2.0/((gamma+1.0)*rhoL)
    AR = 2.0/((gamma+1.0)*rhoR)
    BL = (gamma-1.0) / (gamma+1.0) * PL
    BR = (gamma-1.0) / (gamma+1.0) * PR
    csL = math.sqrt(gamma*PL/rhoL)
    csR = math.sqrt(gamma*PR/rhoR)

    p1 = 0.5*(PL + PR)
    p = p1
    i = 0
    dp = np.inf
    while abs(dp) > abs(p*TOL) and i < MAX:
        p = p1
        f1, df1 = riemann_f(p, rhoL, vL, PL, gamma, AL, BL, csL)
        f2, df2 = riemann_f(p, rhoR, vR, PR, gamma, AR, BR, csR)
        f = f1 + f2 + vR-vL
        df = df1 + df2

        dp = -f/df
        p1 = p + dp
        i += 1

    p = p1
    u = 0.5*(vL+vR) + 0.5*(riemann_f(p, rhoR, vR, PR, gamma, AR, BR, csR)[0]
                - riemann_f(p, rhoL, vL, PL, gamma, AL, BL, csL)[0])

    X = a + ((b-a)/N) * (np.arange(N) + 0.5)
    xi = X/T

    rho = np.empty(X.shape)
    v = np.empty(X.shape)
    P = np.empty(X.shape)

    if p > PL:
        # Left Shock
        rhoLS = rhoL * (p/PL + (gamma-1.0)/(gamma+1.0)) / (
                (gamma-1.0)/(gamma+1.0) * p/PL + 1.0)
        SL = vL - csL*math.sqrt((gamma+1) * p/PL + (gamma-1)/(2*gamma))

        iL = xi < SL
        iLS = (xi >= SL) * (xi < u)
        rho[iL] = rhoL
        v[iL] = vL
        P[iL] = PL
        rho[iLS] = rhoLS
        v[iLS] = u
        P[iLS] = p
    else:
        # Left Rarefaction
        rhoLS = rhoL * math.pow(p/PL, 1.0/gamma)
        csLS = csL * math.pow(p/PL, (gamma-1.0) / (2*gamma))
        SHL = vL - csL
        STL = u - csLS
        
        iL = xi < SHL
        ifan = (xi >= SHL) * (xi < STL)
        iLS = (xi >= STL)*(xi < u) 

        rho[iL] = rhoL
        v[iL] = vL
        P[iL] = PL
        rho[ifan] = rhoL * np.power(2.0/(gamma+1) + (gamma-1)/(gamma+1) 
                                    * (vL - xi[ifan]) / csL, 2.0/(gamma-1.0))
        v[ifan] = 2.0/(gamma+1) * (csL + 0.5*(gamma-1)*vL + xi[ifan])
        P[ifan] = PL * np.power(2.0/(gamma+1) + (gamma-1)/(gamma+1) 
                                * (vL - xi[ifan]) / csL, 2.0*gamma/(gamma-1.0))
        rho[iLS] = rhoLS
        v[iLS] = u
        P[iLS] = p

    if p > PR:
        # Right Shock
        rhoRS = rhoR * (p/PR + (gamma-1.0)/(gamma+1.0)) / (
                (gamma-1.0)/(gamma+1.0) * p/PR + 1.0)
        SR = vR + csR*math.sqrt((gamma+1) * p/PR + (gamma-1)/(2*gamma))

        iR = xi >= SR
        iRS = (xi < SR) * (xi >= u)
        rho[iR] = rhoR
        v[iR] = vR
        P[iR] = PR
        rho[iRS] = rhoRS
        v[iRS] = u
        P[iRS] = p
    else:
        # Right Rarefaction
        rhoRS = rhoR * math.pow(p/PR, 1.0/gamma)
        csRS = csR * math.pow(p/PR, (gamma-1.0) / (2*gamma))
        SHR = vR + csR
        STR = u + csRS
        
        iR = xi >= SHR
        ifan = (xi < SHR) * (xi >= STR)
        iRS = (xi < STR)*(x >= u) 

        rho[iR] = rhoR
        v[iR] = vR
        P[iR] = PR
        rho[ifan] = rhoR * np.power(2.0/(gamma+1) - (gamma-1)/(gamma+1) 
                                    * (vR - xi[ifan]) / csR, 2.0/(gamma-1.0))
        v[ifan] = 2.0/(gamma+1) * (-csR + 0.5*(gamma-1)*vR + xi[ifan])
        P[ifan] = PR * np.power(2.0/(gamma+1) - (gamma-1)/(gamma+1) 
                                * (vR - xi[ifan]) / csR, 2.0*gamma/(gamma-1.0))
        rho[iRS] = rhoRS
        v[iRS] = u
        P[iRS] = p

    return X, rho, v, P

if __name__ == "__main__":

    rhoL = 1.0
    PL = 1.0
    vL = 0.0
    rhoR = 0.125
    PR = 0.1
    vR = 0.0
    gamma = 1.4

    X, rho, v, P = riemann(-1.0, 1.0, 0.0, 1000, 0.3, rhoL, vL, PL,
                            rhoR, vR, PR, gamma)

    import matplotlib.pyplot as plt
    
    AL = 2.0/((gamma+1.0)*rhoL)
    AR = 2.0/((gamma+1.0)*rhoR)
    BL = (gamma-1.0) / (gamma+1.0) * PL
    BR = (gamma-1.0) / (gamma+1.0) * PR
    csL = math.sqrt(gamma*PL/rhoL)
    csR = math.sqrt(gamma*PR/rhoR)
    
    fig1, ax1 = plt.subplots(2,1)
    PP = np.logspace(-2,2,100)
    FL = [riemann_f(p,rhoL, vL, PL, gamma, AL, BL, csL)[0] for p in PP]
    dFL = [riemann_f(p,rhoL, vL, PL, gamma, AL, BL, csL)[1] for p in PP]
    FR = [riemann_f(p,rhoR, vR, PR, gamma, AR, BR, csR)[0] for p in PP]
    dFR = [riemann_f(p,rhoR, vR, PR, gamma, AR, BR, csR)[1] for p in PP]
    ax1[0].plot(PP, FL)
    ax1[0].plot(PP, FR)
    ax1[1].plot(PP, dFL)
    ax1[1].plot(PP, dFR)
    ax1[0].set_xscale('log')
    ax1[1].set_xscale('log')

    fig, ax = plt.subplots(3,1)
    ax[0].plot(X, rho)
    ax[1].plot(X, v)
    ax[2].plot(X, P)

    plt.show()

