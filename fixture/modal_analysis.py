from numpy import *
from scipy import interpolate
from scipy.linalg import toeplitz,pinv,inv
from scipy.signal import invres,step,impulse, residue
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
        # Needed 10 for clock feedthrough, but many for other thing...
        CTLE_CUTOFF = 10
        t_interp, h_interp = t_norepeat[CTLE_CUTOFF:], h_norepeat[CTLE_CUTOFF:]

        #h_impulse = diff(h_interp)/diff(t_interp) # get impulse response from step response
        t_impulse = t_interp[:-1]+diff(t_interp)/2.0 # time adjustment, take the mid-point



        last_interesting_index = where(abs(h_impulse) > max(abs(h_impulse))/1000)[0][-1]
        first_good_index = argmax(abs(h_impulse))
        h_impulse_crop = h_impulse[first_good_index:last_interesting_index]
        t_impulse_crop = t_impulse[first_good_index:last_interesting_index]

        #no_sample_impulse = 10
        #spline_fn = interpolate.UnivariateSpline(t_impulse_crop, h_impulse_crop, k=3)#, s=len(h_impulse_crop)**2*1e9)
        ## I believe the smoothing factor is the maximum MSE of the re-done points
        ## so we want higher numbers so it can be more lax
        #spline_fn.set_smoothing_factor((1e7)**2*len(t_impulse_crop))
        #t_impulse_interp = linspace(t_impulse_crop[0], t_impulse_crop[-1], no_sample_impulse)
        ##t_impulse_interp = t_impulse_interp[1:-1]
        #h_impulse_interp = spline_fn(t_impulse_interp)


        if self.debug:
            plt.plot(t, h, '-+')
            plt.plot(t_interp, h_interp, '-+')
            plt.grid()
            plt.show()

            plt.plot(t_impulse, h_impulse, '+')
            plt.plot(t_impulse_crop, h_impulse_crop)
            #plt.plot(t_impulse_interp, h_impulse_interp, '-+')
            plt.grid()
            plt.show()

        #f = open('temp_impulse_response.pickle', 'wb')
        #pickle.dump((t_impulse, h_impulse), f)
        #f.close()

        #return self.fit_impulseresponse(h_impulse_interp,t_impulse_interp)
        return self.fit_impulseresponse(h_impulse_crop, t_impulse_crop)
    
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

        # STEP 1: Extract N poles. This is the difficult step
        # Goal: if h looks like a superposition of N c*e^(wt) components,
        # what are the N values of w ?
        # Take sets of (N+1) evenly-spaced datapoints and think of them
        # like a recurrence relation
        # h[n] = a1*h[n-1] + a2*h[n-2]
        # So pack h[n-1] and h[n-1] into columns of R, where n is the row
        R  = matrix(toeplitz(h[N-1::-1],h[N-1:no_sample-1]))

        # Consider again h[n] = a1*h[n-1] + a2*h[n-2]
        # R*A represents the right side, h (slightly cropped) represents left
        # We left-multiply both sides by the pseudo-inverse of R to solve for
        # the best-fit version of A. The entries of A are like a1 and a2
        # Ignore the negation for now, it will make more sense in the next step
        A = -1*matrix(pinv(R.transpose()))*matrix(h[N+arange(0,M,dtype=int)])

        # Consider again h[n] = a1*h[n-1] + a2*h[n-2]
        # Move the right hand side to the left
        # h[n] - a1*h[n-1] - a2*h[n-2] = 0
        # Guess h[n] = c*r^n
        # c*r^(n-1)(1*r^2 - a1*r - a2) = 0
        # We see A0 represents the coefficients of that polynomial in r
        A0 = vstack((ones(A.shape[1]),A)).getA1()

        # Roots of this polynomial are the acceptable values of r
        # We are looking for r (and eventually c) such that  h[n] = c*r^n
        # Sometimes the roots are complex; we cast as complex64 always because
        # we had issues with taking the log of a negative number when the
        # dtype was not complex
        A0_roots = array(roots(A0), dtype = complex64)

        # Recall we are looking at h[n] = c*r^n, and we just found possible r
        # What we really want is h[n] = c*e^(wt); notice w = ln(r), t = dT*n
        # Unlike math.log and np.log, scimath.log is fine with log(-5)
        import numpy as np
        log = np.lib.scimath.log
        P = matrix(log(A0_roots) / dT).transpose()

        # STEP 2: We have the poles (speeds of decay), now find magnitudes
        # Basic strategy is just linear regression on the magnitude of each
        # frequency component
        # Rows of Q are exponential decays at various frequencies
        Q = exp(P * t.transpose())

        # Do linear regression to find h as a linear combination of rows of Q
        # Z is a coefficient for each row of Q
        # NOTE Z is NOT the zeros. It's the coefficients for each 1/(s-p) term
        # in the Laplace transform
        Z = pinv(matrix(Q.transpose()))*matrix(h)

        # STEP 3: use magnitudes of components to find the zeros
        # Go from A/(s-p1) + B/(s-p2) to rational polynomial coefficients
        # invres finction: calculate b(s) and a(s) from partial fraction
        # expansion, numerators are Z[i], denominators are (s-P[i])
        # TODO not sure about tol here, I think it should be 0 since we never
        # generate special terms for repeated zeros
        num,den = invres(Z.getA1(),P.getA1(),zeros(size(P)),tol=1e-4,rtype='avg')
        error = False
        if not error:
            print(num, den)
        num = num.real
        den = den.real

        h_estimated = Q.transpose()*Z
        h_estimated = h_estimated.real.getA1()
        return dict(h_impulse=h,h_estimated=h_estimated,num=num,den=den,failed=error)

    def constrained_regression(self, A, B, C, D):
        import numpy as np
        # Find matrix X that minimizes AX-B, under the constraint CX=D
        # N is # datapoints, M is (constrained) x dimension, p is # constraints
        n = A.shape[0]
        m = A.shape[1]
        p = C.shape[0]

        # Step 1: find E, F such that E*X_tilde+F always produces valid X
        # fill in rows of C with orthogonal vectors until it's square
        # TODO make sure these are orthogonal ... but they probably are
        C_extension = np.random.rand(m-p, m)
        C_tilde = np.vstack((C, C_extension))
        C_tilde_inv = np.linalg.inv(C_tilde)
        E = C_tilde_inv[:, p:]
        F = C_tilde_inv[:, :p] @ D

        # Step 2: Use X=E*X_tilde+F to re-frame linear regression for X_tilde
        # A (E*X_tilde + F) = B
        # A E X_tilde = B - A F_tilde
        A_tilde = A @ E
        B_tilde = B - (A @ F)

        # Step 3: solve linear regression in tilde space
        X_tilde = pinv(A_tilde) @ B_tilde

        # Step 4: Use E and F to get back from tilde space
        X = E @ X_tilde + F
        return X

    def fit_step_response_direct(self, t, h, NP, NZ):
        import numpy as np
        no_sample = len(t)
        # TODO float equality?
        if np.diff(t).max() > np.diff(t).min() or t[0] < 0:
            print('RESAMPLING')
            spline_fn = interpolate.interp1d(t,h)
            #t = linspace(t[0],t[-1],no_sample)
            # TODO adjust no_sample based on how much of t is before 0?
            t = np.linspace(0,t[-1],no_sample)
            h = spline_fn(t)


        if self.debug:
            import matplotlib.pyplot as plt
            plt.plot(t, h, '-*')
            plt.legend(('Step response after resampling',))
            plt.show()

        h = h.reshape(no_sample,1)
        # TODO step response always starts at 0. But does this fix
        # mess with scaline and dc gain?
        #h -= h[0]
        t = t.reshape(no_sample,1)

        # TODO stride
        desired_dT = (t[-1] - t[0]) / 10
        stride = max(1, int(np.round(desired_dT / (t[1] - t[0]))))
        dT = t[stride] - t[0]
        M = no_sample - NP # no of equations

        # STEP 1: Extract N poles. This is the difficult step
        # Goal: if h looks like a superposition of N c*e^(wt) components,
        # what are the N values of w ?
        # Take sets of (N+1) evenly-spaced datapoints and think of them
        # like a recurrence relation
        # h[n] = a1*h[n-1] + a2*h[n-2]
        # So pack h[n-1] and h[n-1] into columns of R, where n is the row
        #R  = np.matrix(toeplitz(h[N-1::-1], h[N-1:no_sample-1]))

        # Consider again h[n] = a1*h[n-1] + a2*h[n-2]
        # R*A represents the right side, h (slightly cropped) represents left
        # We left-multiply both sides by the pseudo-inverse of R to solve for
        # the best-fit version of A. The entries of A are like a1 and a2
        # Ignore the negation for now, it will make more sense in the next step
        #A = -1*matrix(pinv(R.transpose()))*matrix(h[N+arange(0,M,dtype=int)])

        # Consider again h[n] = a1*h[n-1] + a2*h[n-2] + a3*h[n-3]
        # Move the right hand side to the left
        # h[n] - a1*h[n-1] - a2*h[n-2] - a3*h[n-2] = 0
        # Guess h[n] = c*r^n
        # c*r^(n-1)(1*r^3 - a1*r^2 - a2*r - a3) = 0
        # Because h is a step response, we know r=1 is a solution, so we want
        # 1-a1-a2-a3=0. Let's represent a_desired with b
        # 1*h[n] - a1*h[n-1] - a2*h[n-2] - (-a1 -a2 + 1)*h[n-3] = 0
        # a1*(-h[n-1]+h[n-3]) + a2*(-h[n-2]+h[n-3]) = (-h[n] + h[n-3])


        #A0 = vstack((ones(A.shape[1]),A)).getA1()

        m = no_sample - (NP+1) * stride
        def data(start):
            return h[start:start+m].reshape((m,))

        # AX = B
        A = np.stack(
            (-data((1+i)*stride) + data(0) for i in range(NP-1, -1, -1)),
            1
        )
        B = (-data((1+NP)*stride) + data(0)).reshape((m,1))
        X = np.linalg.pinv(A) @ B
        X_coeffs = np.concatenate(([1], -X.reshape((NP,)), [sum(X)-1]))
        roots = np.roots(X_coeffs)




        # Roots of this polynomial are the acceptable values of r
        # We are looking for r (and eventually c) such that  h[n] = c*r^n
        # Sometimes the roots are complex; we cast as complex64 always because
        # we had issues with taking the log of a negative number when the
        # dtype was not complex
        #A0_roots = array(roots(A0), dtype = complex64)

        # Recall we are looking at h[n] = c*r^n, and we just found possible r
        # What we really want is h[n] = c*e^(wt); notice w = ln(r), t = dT*n
        # Unlike math.log and np.log, scimath.log is fine with log(-5)
        import numpy as np
        log = np.lib.scimath.log
        P_step = np.matrix(log(roots) / dT).transpose()

        # STEP 2: We have the poles (speeds of decay), now find magnitudes
        # Basic strategy is just linear regression on the magnitude of each
        # frequency component
        # Rows of Q are exponential decays at various frequencies
        # NZ makes this a little more difficult: we want to find a magnitude
        # for each of these things, but we require that when we do the inverse
        # residual calculation, the resulting numerator polynomial has a degree
        # of only NZ, which is likely smaller than what it wants
        Q_step = np.exp(P_step * t.transpose())

        # Do linear regression to find h as a linear combination of rows of Q
        # Z is a coefficient for each row of Q
        # NOTE Z is NOT the zeros. It's the coefficients for each 1/(s-p) term
        # in the Laplace transform
        # TODO fewer zeros

        Z_step = self.constrained_regression(Q_step.transpose(),
                                             h,
                                             np.ones((1, NP+1)),
                                             np.array([0]).reshape((1,1)))

        print('Z_step', Z_step)
        print('P_step', P_step)


        # STEP 3: use magnitudes of components to find the zeros
        # Go from A/(s-p1) + B/(s-p2) to rational polynomial coefficients
        # invres finction: calculate b(s) and a(s) from partial fraction
        # expansion, numerators are Z[i], denominators are (s-P[i])
        # TODO not sure about tol here, I think it should be 0 since we never
        # generate special terms for repeated zeros

        num,den = invres(Z_step.getA1(),P_step.getA1(),
                         zeros(size(P_step)),tol=1e-4,rtype='avg')


        print('numerator', num)

        # for a step response, we should have N poles plus a pole at 0,
        # and N-1 zeros. This means our num polynomial should be a degree
        # lower than usual for this many poles, i.e. leading coeff is 0
        #assert len(num) == len(den)-2 or abs(num[0]) < 1e-10
        if len(num) != len(den)-2:
            num = np.delete(num, 0)

        # now factor the top and bottom
        zs_step = np.roots(num)
        ps_step = np.roots(den)


        # remove the pole at 0 and the highest zero
        index_integrator_pole = np.argmin(abs(ps_step))
        P_impulse = np.delete(ps_step, index_integrator_pole).reshape((NP,))
        #index_large_zero = np.argmax(abs(zs_step))
        #Z_impulse = np.delete(zs_step, index_large_zero).reshape((N-1,))
        Z_impulse = zs_step

        # we're done, but for debug we want to produce the time-domain again
        # and we will compare to the step response, so add the pole at 0
        num_test = np.poly(Z_impulse)
        P_impulse_integrated = np.concatenate((P_impulse, [0]))
        den_test = np.poly(P_impulse_integrated)
        res_test, p_test, _ = residue(num_test, den_test)
        # q_test * exp(den_test * t), with dot product & outer product
        #h_test = (q_test.reshape((1,N-1)) @
        #          np.exp(den_test.reshape((N-1,1)),
        #                 t.reshape((1, len(t)))))
        h_test = np.dot(res_test, np.exp(np.outer(p_test, t)))
        # TODO this will break when DC gain is zero
        # TODO also the index should be the index of the pole at 0,
        # which is not always index 0
        scaling_test = (Z_step[0] / res_test[0]).item()
        print('poles for test', p_test)
        print('residue for test', res_test)
        print('gain for test', scaling_test)
        h_test = h_test * scaling_test
        h_test = h_test.real

        h_estimated = Q_step.transpose() * Z_step
        h_estimated = h_estimated.real.getA1()
        if self.debug:
            plt.plot(t, h, 'o')
            plt.plot(t, h_estimated, '--+')
            plt.plot(t, h_test, '-+')
            plt.grid()
            plt.show()

        num = num.real
        den = den.real



        num, den = np.poly(Z_impulse).real, np.poly(P_impulse).real
        return dict(h_impulse=h,h_estimated=h_estimated,num=num,den=den)

    def optimize(self, t, h_step, ps, zs):
        import numpy as np
        import scipy
        # Use a nonlinear optimizer
        # USE HERTZ FOR ps AND zs
        # a little weird that h is step response, ps and zs are impulse
        # H(t) = gain*zs/ps
        Nps = len(ps)
        Nzs = len(zs)

        ps = np.array(ps) * (2*np.pi)
        zs = np.array(zs) * (2*np.pi)


        def time(xs):
            ps = xs[:Nps]

            zs = xs[Nps:Nps + Nzs]
            gain = xs[Nps + Nzs]
            ps_impulse = np.concatenate(([0], ps))  # integrate
            num = gain * np.poly(zs)
            den = np.poly(ps_impulse)
            rs, ps_impulse, ks = scipy.signal.residue(num, den, tol=1e-3)
            h_step_time = np.zeros(h_step.shape, dtype=ps_impulse.dtype)
            for r, p in zip(rs, ps_impulse):
                h_step_time += r * np.exp(p * t)
            return h_step_time

        def err(xs):
            h_step_time = time(xs)
            err_vec = h_step - h_step_time
            # TODO weight err_vec by dt
            error = sum(abs(err_vec ** 2))
            #print('got error', error, xs[:3] , xs[3])
            return error

        x0_nogain = np.concatenate((ps, zs, [1]))
        h_step_nogain = time(x0_nogain)
        gain = h_step[-1] / h_step_nogain[-1]

        x0 = np.concatenate((ps, zs, [gain]))
        result = scipy.optimize.minimize(err, x0, options={'disp': True, 'eps': 1e-7, 'gtol': 1e-20, 'xatol': 1e6, 'fatol': 1e-6})
        x_fit = result.x
        h_step_fit = time(x_fit)
        h_step_0 = time(x0)
        x_cheat = np.concatenate((2*np.pi*np.array([-4e9, -5e9, -4.8e9]), [gain]))
        h_step_cheat = time(x_cheat)


        if self.debug:
            import matplotlib.pyplot as plt
            plt.plot(t, h_step, 'o')
            plt.plot(t, h_step_0, '--+')
            plt.plot(t, h_step_fit)
            plt.plot(t, h_step_cheat, '*')
            plt.grid()
            plt.legend(('h_step', 'h_step_0', 'h_step_fit', 'h_step_cheat'))
            plt.show()

        ps_fit = x_fit[:Nps]
        zs_fit = x_fit[Nps:Nps+Nzs]
        return ps_fit, zs_fit



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
