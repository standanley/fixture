from numpy import *
from scipy import interpolate
from scipy.linalg import toeplitz,pinv,inv
from scipy.signal import invres,step,impulse
import pickle
import matplotlib.pyplot as plt

from fixture import template_creation_utils


class ModalAnalysis(object):
    debug = False
    ''' Fit an step response measurement to a linear system model '''
    def __init__(self, rho_threshold = 0.999, N_degree = 50):
        ''' set constraints on calculation '''
        self.rho_threshold = rho_threshold # correlation threshold
        self.N_degree = N_degree # Maximum degree of freedom

    def fit_stepresponse(self,h,t):
        ''' fit an step response measurement to a linear system model '''



        # issues with multiple repeated ts
        t_norepeat, h_norepeat = [], []
        tt_prev= float('nan')
        for tt, hh in zip(t, h):
            if tt != tt_prev:
                t_norepeat.append(tt)
                h_norepeat.append(hh)
                tt_prev = tt
            else:
                # take last version
                h_norepeat[-1] = hh


        no_sample = len(t_norepeat)

        #spline_fn = interpolate.InterpolatedUnivariateSpline(t_norepeat, h_norepeat)
        ## t = linspace(t[0],t[-1],no_sample)
        ## TODO i intentionally cut off the first point below to see if it would fix problems but it did not
        ## if the problems go away, we should try putting that point back
        #t_interp = linspace(t_norepeat[0], t_norepeat[-1], no_sample)
        ##t_interp = t_interp[1:-1]
        #h_interp = spline_fn(t_interp)
        # I DO want to cut the first 2 off here, otherwise derivative is weird
        # First is in case fault gave us one time step early, second because
        # derivative is still settling
        t_interp, h_interp = t_norepeat[2:], h_norepeat[2:]

        # this is temporary, for the ctle specifically
        CTLE_CUTOFF = 10
        t_interp, h_interp = t_norepeat[CTLE_CUTOFF:], h_norepeat[CTLE_CUTOFF:]

        h_impulse = diff(h_interp)/diff(t_interp) # get impulse response from step response
        t_impulse = t_interp[:-1]+diff(t_interp)/2.0 # time adjustment, take the mid-point



        last_interesting_index = where(abs(h_impulse) > max(abs(h_impulse))/1000)[0][-1]
        h_impulse_crop = h_impulse[:last_interesting_index]
        t_impulse_crop = t_impulse[:last_interesting_index]

        no_sample_impulse = 10
        spline_fn = interpolate.UnivariateSpline(t_impulse_crop, h_impulse_crop, k=3)#, s=len(h_impulse_crop)**2*1e9)
        # I believe the smoothing factor is the maximum MSE of the re-done points
        # so we want higher numbers so it can be more lax
        spline_fn.set_smoothing_factor((1e7)**2*len(t_impulse_crop))
        t_impulse_interp = linspace(t_impulse_crop[0], t_impulse_crop[-1], no_sample_impulse)
        #t_impulse_interp = t_impulse_interp[1:-1]
        h_impulse_interp = spline_fn(t_impulse_interp)


        if self.debug:
            plt.plot(t, h, '-+')
            plt.plot(t_interp, h_interp, '-+')
            plt.grid()
            plt.show()

            plt.plot(t_impulse, h_impulse, '+')
            plt.plot(t_impulse_crop, h_impulse_crop)
            plt.plot(t_impulse_interp, h_impulse_interp, '-+')
            plt.grid()
            plt.show()

        f = open('temp_impulse_response.pickle', 'wb')
        pickle.dump((t_impulse, h_impulse), f)
        f.close()

        return self.fit_impulseresponse(h_impulse_interp,t_impulse_interp)
    
    def fit_impulseresponse(self,h,t):
        #for N in range(2,self.N_degree):
        for N in range(1,self.N_degree+1):
            print('Trying degree', N)
            ls_result = self.leastsquare_complexexponential(h,t,N)
            if ls_result['failed'] and N >=3:
                print('Giving up because degree', N, 'failed')
                break
            rho = self.compare_impulseresponse(ls_result['h_impulse'],ls_result['h_estimated'])
            print('rho=',rho)
            if rho > self.rho_threshold:
                print('Happy with this rho, moving on now')
                break
            if N == self.N_degree:
                print('[WARNING]: Maximum degree of freedom is reached when fitting response to transfer function')
        return ls_result

    def compare_impulseresponse(self,h1,h2):

        if self.debug:
            import matplotlib.pyplot as plt
            print('comparing impulse responses. x axis has been lost; just index now')
            plt.plot(h1, '-*')
            plt.plot(h2, '-*')
            plt.show()


        h1 = h1.flatten()
        h2 = h2.flatten()
        return corrcoef([h1,h2])[0,1]

    def leastsquare_complexexponential(self,h,t,N):
        ''' Model parameter estimation from impulse response
                using least-squares complex exponential method 
                h: impulse response measurement
                t: time range vector (in sec); must be uniformly-spaced
                N: degree of freedom
        '''


        # TODO this was added 5/3/21 when I was having trouble with 2-poles
        # seemed to make no difference
        t = t - t[0]


        h_temp, t_temp = h, t
        no_sample = h.size
        if False: #diff(t).max() > diff(t).min(): # check for uniform time steps
            #no_sample //= 250
            no_sample = 20
            print('RESAMPLING')
            spline_fn = interpolate.InterpolatedUnivariateSpline(t,h)
            #t = linspace(t[0],t[-1],no_sample)
            # TODO i intentionally cut off the first point below to see if it would fix problems but it did not
            # if the problems go away, we should try putting that point back
            t = linspace(t[0],t[-1],no_sample)
            #t = t[1:]
            h = spline_fn(t)
            #h = h / abs(max(h))

            if self.debug:
                import matplotlib.pyplot as plt
                #plt.plot(t_temp, h_temp / abs(max(h_temp)), '-+')
                plt.plot(t_temp, h_temp, '-+')
                plt.plot(t, h, '-+')
                plt.legend(['Original', 'Resampled'])
                plt.grid()
                plt.show()

        #import matplotlib.pyplot as plt
        #plt.plot(h, '-*')
        #plt.show()




        h = h.reshape(no_sample,1)
        t = t.reshape(no_sample,1)
        M = no_sample - N # no of equations
        dT = t[1]-t[0]
        # least-squares estimation of eigenvalues (modes)

        # Sizes assuming no_sample=100, N=2

        # R is matrix version of "convolve with impulse response"
        # R is 2x98, which is N x M
        # NOTE why is it no_sample-1 here? I feel like it skips the last sample...
        R  = matrix(toeplitz(h[N-1::-1],h[N-1:no_sample-1]))

        # pseudoinverse of R_T is 2x98
        # N+arange(...) = [2, 3, 4, ... 99], so 98 total elements
        # (A 2x1) = (R_T_inv 2x98) * (h 98x1)
        # I expected it to do element-wise multiplicaiton, but it did matrix

        # my interpretation:
        # R_T: map from 2 samples to the system response to those 2 values and a bunch of zeros
        # pinv(R_T): simplest possible 98 -> 2 mapping that inverts R_T
        # A: put h into pinv(R_T). That doesn't make much conceptual sense to me...
        A = -1*matrix(pinv(R.transpose()))*matrix(h[N+arange(0,M,dtype=int)])

        # A0.shape = (3,) so it's just a vector?
        # vstack(...) is just 3x1
        # looks like .getA1 is defined in defmatrix.py, it just flattens
        # A0 is just A but with a 1 prepended ... i.e. A=[[-2], [.5]] -> A0=[1, -2, .5]
        A0 = vstack((ones(A.shape[1]),A)).getA1()

        # as expected, A0_roots is length 2
        A0_roots = roots(A0)
        #A0_roots = [(0 if r < 0 and r > -1e-9 else r) for r in A0_roots]


        ''' NOTE from my musings: 
        Take -A to be the coefficients of a recurrence relation, i.e.
        x[n] = sum(x[n-i]*A[i]), possibly with i negative or reversed...
        Then we guess x[n] = r^n
        A0_roots will give us the possible values of r
        
        ALSO coming from the other side
        We know P are coefficients like h[t] = e^(t*P[i])
        P[i] = log(A0_roots[i])/dT  ->  e^(P[i]*dT) = A0_roots[i]
        Can think of this as the amount the signal decays in one dT?
        
        NEXT STEP
        What is the recurrence relation s.t. possible values of r are decay
            coefficients in one dT?
        Can we cast convolution with impulse response as a recurrence relation?
            Of course: out[t] = in[t-1]*h[t-1] + ...
        OOH this looks like a recurrence relation when out=in, i.e. eigenvectors?
        
        REASONING THROUGH
        So if we find a vector x s.t. conv(x, h) = x we can view the convolution
            series as a recurrence relation with coefficients h (we do need to
            be a bit careful about the time delay, but I think the exact
            toeplitz formulation gives us control over that?)
        Now we guess that the eigenvector is an exponential because that lets us
            turn the right side of the convolution into a polynomial in new var r
        Find the roots of this polynomial to find possible values of r, which are
            bases of the exponential, which can be rewritten into factors of the
            exponent (the way we usually see it in step responses)
            
        QUESTIONS
        1) why do we make A at all? I feel like we could use h directly
            I think it has to do with too much data, we want a least squares,
            and this is what the pseudoinverse does for us
        2) in my reasoning through we find x to be an eigenvector but we 
            immediately throw x away by assuming it's an exponential
            I think maybe that's fine? since we don't start by finding eigenvectors
            
            
        Rephrasing it again:
        assume x[t]=e^at is an eigenvector = a mode = something scaled by LTI
        let r=e^a  so that r^t=e^at
        Write out convolution as x[t] = h[t-1]x[t-1] + h[t-2]x[t-2] + ...
        NOTE number of modes you find will be number of terms above. If you
            Only want 2 poles, use pseudoinverse magic to condense h to 2 things?
        Do some algebra to get r^(-N)(r^N - h[t-1]r^(N-1) - ... - h[0])=0
        Now find the roots of that polynomial to see the possible values of r
        a = log(r)
        
        Nice! this is exactly the matrix P
        
        From here it's standard linear algebra:
        Just generate time domain response of each e^at mode, say that h[t]
        is a linear combination of these, find the coefficients
        '''


        # P is a 2x1 matrix
        import numpy as np
        A0_roots = np.array(A0_roots, dtype = np.complex64)
        # unlike math.log and np.log, this is fine with log(-5)
        log = np.lib.scimath.log
        P = matrix(log(A0_roots) / dT).transpose()

        # least-squares estimation of eigenvectors (modal coef)
        # q is 2x99?
        Q = exp(P * t.transpose())

        error = False
        try:
            Z = pinv(matrix(Q.transpose()))*matrix(h)
            print(type(Z))
        except ValueError:
            error = True
            print('Error calculating poles/zeros')

            import matplotlib.pyplot as plt
            plt.plot(t, h, '-*')
            plt.show()

            Z = matrix(zeros((Q.shape[0],h.shape[1])))
        # return values


        # invres is: calculate b(s) and a(s) from partial fraction expansion
        # numerators are Z[i], denominators are (s-P[i])
        num,den = invres(Z.getA1(),P.getA1(),zeros(size(P)),tol=1e-4,rtype='avg')
        if not error:
            print(num, den)
        num = num.real
        den = den.real
        h_estimated = Q.transpose()*Z
        h_estimated = h_estimated.real.getA1()
        return dict(h_impulse=h,h_estimated=h_estimated,num=num,den=den,failed=error)

    def fit_transferfunction(self,H,f):
        ''' Fit a frequency response measurement to a linear system model '''
        pass

    def compare_transferfunction(self,H1,H2):
        ''' calculates the correlation coef. in frequency domain between two transfer function '''
        pass

    def selftest(self):
        pass

def main():
    import matplotlib.pyplot as plt
    X = ModalAnalysis(0.999,100)
    system = ([2.0],[1.0,2.0,1.0])
    #system = ([2.0,1.0],[1.0,2.0,1.0])
    ## impulse response test
    #t,h = impulse(system)
    #ls_result=X.fit_impulseresponse(h,t)
    #system_est = (ls_result['num'],ls_result['den'])
    #t1,h1 = impulse(system_est)
    #plt.plot(t,h,'rs',t1,h1,'bx')
    #plt.show()
    ## step response test
    t,h = step(system)
    ls_result=X.fit_stepresponse(h,t)
    system_est = (ls_result['num'],ls_result['den'])
    t1,h1 = step(system_est)
    plt.plot(t,h,'rs',t1,h1,'bx')
    plt.show()

if __name__ == "__main__":
    main()
