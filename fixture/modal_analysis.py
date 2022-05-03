import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
DEBUG = True
from scipy import interpolate

class ModalAnalysis:
    debug = True

    def get_test_h_step(self):
        np.random.seed(5)
        N = 1000
        dt = 1e-12
        t = dt*np.array(range(N)) # N seconds
        ps = np.array([0, -4e9, -6e9, -10e9])
        #ps = np.array([0, -2e9+5e9j, -2e9-5e9j, -3e9, 10e9])
        rs = np.exp(ps * dt)
        NP = len(ps)


        cs = np.random.normal(0, 2, rs.shape)
        #cs = np.array([42, -1, -1, 2.5, 42])

        # remove second zero (must also include remove one zero line)
        cs[-1] = -sum(cs[1:-1] * ps[1:-1]) / ps[-1]
        # remove one zero
        cs[0] = -sum(cs[1:])

        print(cs)
        h_step_noiseless = np.real(np.array([sum(cs*rs**(n/dt)) for n in t]))
        h_step = h_step_noiseless + np.random.normal(0, 0.00001, h_step_noiseless.shape)

        if self.debug:
            plt.figure()
            plt.plot(t, h_step, '+-')
            plt.grid()
            plt.show()
        return h_step


    def get_slowest_pole(self, h_step, t, NP, known_poles, target_period):
        # we assume the data starts at t=0; we accept earlier starts for the
        # purpose of interpolating the value at t=0
        #print('\nget_slowest called with', NP, known_poles, stride)

        interp = interpolate.interp1d(t, h_step)
        # We want NUM_DATA rows, each row corresponds to target_period
        # seconds, and we only look at data up to 5 total periods after t=0
        # If there's not enough h_step data then squish, don't truncate
        NUM_DATA = 100
        total_time = t[-1]
        chunk_duration = min(target_period, total_time/2)
        first_chunk_start = min(0.2*target_period, total_time-chunk_duration*2)
        dt = chunk_duration / (NP)
        last_chunk_start = min(target_period * 5, total_time - chunk_duration)
        starts = np.linspace(first_chunk_start, last_chunk_start, NUM_DATA)
        samples = [interp(np.linspace(s, s+chunk_duration, NP+1))
                   for s in starts]
        # make samples rows in a (tall) matrix
        data = np.stack(samples, 0)
        print(f'using data from {first_chunk_start} to {last_chunk_start}, chunk {chunk_duration}, period {target_period}')

        # We split the collected data into the initial points, and the final point
        A = data[:,:-1]
        B = data[:,-1:]

        # Consider 5 poles, 2 already known
        # We can do some linear algebra to find column vector X such that
        # x0*a[n] + x1*a[n+1] + x2*a[n+2] + x3*a[n+3] + x4*a[n+4] + a[n+5] = 0
        # a[n](x0 + x1*r + x2*r^2 + x3*r^3 + x4*r^4 + r^5) = 0
        # r^5 + x4*r^4 + x3*r^3 + x2*r^2 + x1*r + x0 = 0
        # and then solve this polynomial to find the roots

        # BUT
        # With 2 known poles, we know this should factor out to
        # (r-p1)(r-p2)(r^3 + y0*r^2 + y1*r + y2) = 0
        # So we really want to use our linear algebra to find the Y vector instead
        # First we need a matrix Z, which depends on the known ps, s.t. X = ZY, (5x1) = (5x3)(3x1)
        # Then our linear algebra that was AX+B=0 becomes AZY+B=0, which is easily solvable for Y

        # Step 1: find Z
        # A a reminder, X = ZY where X and Y are defined as:
        # r^5 + x4*r^4 + x3*r^3 + x2*r^2 + x1*r + x0 = 0
        # (r-p1)(r-p2)(r^3 + y2*r^2 + y1*r + y0) = 0
        # Define C, which is coefficients of poly from known roots
        # r^2 + (-p1-p2)*r + p1*p2 -> c0=p1*p2, c1=(-p1-p2)
        # We see that each term in X[i] is a product of Y[j] and C[k] terms s.t. i=j+k
        # SO we can directly write Z[i,j] = C[i-j], or 0 if that's outside the C bounds

        # BE CAREFUL about the leading 1s in these polynomials.
        # In our example, the full Z would be 6x4, including leading 1s. It's okay to drop the
        # bottom row because it only contributes to the leading 1 of x, which we want to drop.
        # But we can't drop the right column, which corresponds to the leading 1 of Y, because
        # it contributes to other rows in X.
        # Z: 5x3 version of Z, with bottom row and right column dropped
        # Z~: 5x4 version of Z, with only bottom row dropped
        # Y~: 4x1 version of Y, with a constant one in the fourth spot
        # A Z~ Y~ == -B
        # We can't use least-squares to find Y right now because of that required constant 1
        # E: 4x3 almost-identity
        # F: 4x1 column, [0,0,0,1]
        # A Z~ (E Y + F) == -B
        # A Z~ E Y  +  A Z~ F == -B
        # A Z Y == -B - A Z~_last_column
        # So we need to do a modified regression: we can drop that extra column on Z~, but we have
        # to first use it to modify the B vector
        # Similarly, X = Z~ Y~ becomes X = Z Y + Z~_last_column

        known_rs = np.exp(np.array(known_poles)*dt)
        if np.isinf(known_rs).any():
            # probably got a bad pole in known_poles, should just give up
            return []

        poly = np.polynomial.polynomial
        C = poly.polyfromroots(known_rs)

        Z_tilde = np.zeros((NP, NP-len(known_rs)+1), dtype=C.dtype)
        for i in range(Z_tilde.shape[0]):
            for j in range(Z_tilde.shape[1]):
                k = i-j
                if k >= 0 and k < len(C):
                    Z_tilde[i,j] = C[k]
        Z = Z_tilde[:,:-1]
        Z_column = Z_tilde[:,-1:]

        Y = np.linalg.pinv(A@Z) @ (-B - (A@Z_column))
        X = Z@Y + Z_column

        # x0 * d0 + x1 * d2 + d2 = 0
        # a[n](x0 + x1*r + r^2) = 0

        poly = np.concatenate([[1], X[::-1,0]])

        #print('poly', poly)
        roots = np.roots(poly)
        #print('roots', roots)

        # errors often cause small roots to go negative when they shouldn't.
        # This messes with the log, so we explicitly call those nan.
        def mylog(x):
            if (np.real(x) == 0):
                return float('nan')
            if not abs(np.imag(x)/np.real(x)) > 1e-6 and np.real(x) <= 0:
                return float('nan')
            else:
                return np.log(x)
        ps_with_stride = np.vectorize(mylog)(roots)
        ps = ps_with_stride / (dt)

        # remove known poles
        key=lambda x: float('inf') if np.isnan(x) else abs(x)
        new_ps = sorted(ps, key=key)
        #print('Before processing, found ps', new_ps)
        for known_p in known_poles:
            for i in range(len(new_ps)):
                # TODO is this epsilon reasonable when ps are likely ~1e10?
                if new_ps[i] is not None and abs((new_ps[i] - known_p) < 1e-6):
                    new_ps.pop(i)
                    break
            else:
                if np.isnan(new_ps).any():
                    # the nans are probably causing the error
                    return []
                #print(known_poles)
                #print(new_ps)
                assert False, f'Known pole {known_p} not found!'

        # finally, return 0 (if nan), 1, or 2 (if complex conjugate) slowest new poles
        #print('After processing, new ps', new_ps)
        assert len(new_ps) > 0, 'Found no new poles ... check NP and len(known_poles)'
        if abs(np.imag(new_ps[0])) > 1e-6:
            # complex conjugate pair
            #print(ps, new_ps)
            assert len(new_ps) >= 2, 'Only found one of complex conjugate pair?'
            if abs(np.conj(new_ps[0]) - new_ps[1]) > 1e-6 and np.isnan(new_ps).any():
                return []
            assert abs(np.conj(new_ps[0]) - new_ps[1]) < 1e-6, 'Issue with conjugate pair, check sorting?'
            return new_ps[:2]
        elif not np.isnan(new_ps[0]):
            return new_ps[:1]
        else:
            # empty list
            return new_ps[:0]


    def get_all_poles(self, h_step, t, NP, known_poles):
        known_poles = known_poles.copy()
        # get_largest_pole works well when the stride is correctly chosen for the
        # largest non-given pole
        period_ratio = 2
        target_period = (t[-1] - t[0]) * 0.5
        min_period = t[2]
        while len(known_poles) < NP:
            #stride = max(1, int((target_period/dt)/NP))
            new_ps = self.get_slowest_pole(h_step, t, NP, known_poles, target_period)

            # for complex conjugate, we want to look at the faster of the envelope and ring
            period = (float('inf') if len(new_ps) == 0
                      else abs(1/new_ps[0]) if len(new_ps) == 1
                      else min(abs(1/np.real(new_ps[0])), abs(1/np.imag(new_ps[0]))))

            # TODO I have no idea why this period*=3 used to be here uncommented
            #period *= 3

            # the exponent in period_ratio**2 is kind of empirical
            if period < target_period / period_ratio**2 and target_period > min_period:
                if self.debug:
                    #print('rejecting pole(s)', new_ps, 'because period', target_period, ' was too large')
                    pass
                target_period /= period_ratio
            elif len(new_ps) == 0 and target_period > min_period:
                if self.debug:
                    #print('didnt find new poles; next round would have smaller period')
                    pass
                target_period /= period_ratio
            elif len(new_ps) == 0:
                if self.debug:
                    print('Did not find all the requested poles!')
                break
            else:
                known_poles += list(new_ps)
                if self.debug:
                    print('added pole(s)', new_ps, 'target period now', target_period)
        return np.array(known_poles)

    def get_zeros(self, h_step, dt, poles, NZ):
        #print('get_zeros with ', poles, dt, NZ)
        assert len(set(poles)) == len(poles), 'get_zeros cannot handle repeated roots'
        NP = len(poles)
        #print(NP, poles)
        # If you leave the timescale as 1, you probably get precision issues
        #timescale_dt = 1/(max(abs(poles)) * len(h_step))
        timescale_dt = dt
        poles_column = np.array(poles).reshape((len(poles), 1))
        poles_scaled = poles_column * timescale_dt
        h_step_column = h_step.reshape((len(h_step), 1))
        t = np.array(range(len(h_step))) * timescale_dt
        exps = np.exp(poles_column * t)

        # Normally, we can find optimal coefficients for each row of exps with linear regression
        # But when we are restricted to fewer zeros, we have a problem.
        # We have to ensure that after doing inverse PFD with our coefficients, the leading term(s) in the numerator go to zero
        # c0/(s-r0) + c1/(s-r1) + c2/(s-r2) -> c0(s-r1)(s-r2) + c1(s-r0)(s-r2) + c2(s-r0)(s-r1) = n2*s^2 + n1*s + n0
        # we can find a square matrix Z~ s.t. N~ = Z~ C, where the ~ means we're doing the large version including required zero entries
        # C = (Z~)^-1 N~
        # Because the bottom entries of N~ are required to be zero, we can drop them and the corresponding right columns from (Z~)^-1
        # C = Zinv N    (3x1) = (3x2) (2x1)
        # Now we drop that in our regressino to find the coefficients
        # A C = B           (100x3)(3x1) = (100x1)
        # A (Zinv N) = B    (100x3)(3x2)(2x1) = (100x1)
        # N = (A Zinv)^-1 B
        # And now N is fewer entries, and we can pad the bottom with zeros if we like
        Z_tilde = np.zeros((NP, NP), dtype=poles.dtype)
        poly = np.polynomial.polynomial
        for i in range(NP):
            temp = poly.polyfromroots([poles_scaled[j][0] for j in range(NP) if j != i])
            for k, c in enumerate(temp):
                Z_tilde[k, i] = c

        # I think this should be invertible as long as there are no repeated roots (which is not allowed)
        Z_tilde_inv = np.linalg.inv(Z_tilde)
        Z_inv = Z_tilde_inv[:, :NZ+1]

        #print(exps.shape, Z_inv.shape, h_step.shape)
        N = np.linalg.pinv((exps.T @ Z_inv)) @ h_step_column

        zeros = np.roots(N.reshape(NZ+1)[::-1]) / dt
        #print('zeros', zeros)


        if self.debug:
            h_step_est = exps.T @ X
            h_step_est_2 = exps.T @ coefs
            plt.figure()
            plt.plot(t, h_step, 'o')
            plt.plot(t, np.real(h_step_est))
            plt.plot(t, np.real(h_step_est_2), '--')
            plt.legend(['Orig', 'est', 'est_2'])
            plt.grid()
            plt.title('I forget')
            plt.show()


        dc = N[0] / np.prod([-p*dt for p in poles if p != 0])
        #scale = N[-1]
        zeros_sorted = zeros[np.lexsort((abs(zeros),))]
        return zeros_sorted, dc

    def debug_plot(self, h_step, dt, ps, zs, scale, custom_t=None):
        if custom_t is None:
            t = np.array(range(len(h_step))) * dt
        else:
            t = custom_t

        zs = zs[np.isfinite(zs)]
        ps = ps[np.isfinite(ps)]

        from scipy.signal import residue
        r, p, k = residue(np.poly(zs), np.poly(ps))

        h_step_est = np.zeros(h_step.shape)
        for coef, pole in zip(r, p):
            h_step_est = h_step_est + scale * coef * np.exp(pole*t)
            print('Added coef', scale*coef, 'for pole', pole)

        plt.figure()
        plt.plot(t, h_step, '--+')
        plt.plot(t, h_step_est, '-+')
        plt.legend(['Step response', 'Fit'])
        plt.grid()
        plt.show()
        pass


    def resample(self, h, t):
        # the 5 here is kind of empirical. We want to oversample a bit
        no_sample = len(h)*5
        spline_fn = interpolate.interp1d(t,h)
        t_new = np.linspace(0,t[-1], no_sample)
        dt = (t_new[-1] - t_new[0]) / (len(t_new) - 1)
        h_new = spline_fn(t_new)

        if False and self.debug:
            plt.figure()
            plt.plot(t, h, 'o')
            plt.plot(t_new, h_new, '-+')
            plt.legend(['Original', 'Resampled'])
            plt.grid()
            plt.show()

        return h_new, dt

    def debug_fit_with_poles(self, t, h_step, poles):
        poles_column = np.array(poles).reshape((len(poles), 1))
        # poles_scaled = poles_column * timescale_dt
        exps = np.exp(poles_column * t)

        coefs = np.linalg.pinv(exps.T) @ h_step
        h_step_est = exps.T @ coefs
        return h_step_est

    def extract_pzs_inner(self, h_step, t, NP, NZ, known_poles):
        # first, extract poles with no regard to how many were asked for

        def error_poles(poles):
            poles_column = np.array(poles).reshape((len(poles), 1))
            #poles_scaled = poles_column * timescale_dt
            #t = np.array(range(len(h_step))) * dt
            exps = np.exp(poles_column * t)

            h_step_column = h_step.reshape((len(h_step), 1))
            coefs = np.linalg.pinv(exps.T) @ h_step
            h_step_est = exps.T @ coefs
            err = sum((h_step - h_step_est)**2) / len(h_step)
            return err

        all_results = [[]]
        best_num = 0
        best_err = float('inf')
        num_ps = 1
        while num_ps < NP+3:
            # TODO dt
            poles = self.get_all_poles(h_step, t, num_ps, known_poles)
            #poles = poles * 1/dt

            if self.debug:
                #t = np.array(range(len(h_step))) * dt
                plt.plot(t, self.debug_fit_with_poles(t, h_step, poles))

            err = error_poles(poles)
            all_results.append(poles)
            #print(num_ps, -np.log(abs(err)), poles)
            if len(poles) > len(all_results[best_num]) and err < best_err * .5:
                #print('NEW BEST', num_ps, len(poles))
                best_num = len(all_results)-1
                best_err = err
            num_ps += 1

        if self.debug:
            #t = np.array(range(len(h_step))) * dt
            plt.plot(t, h_step, 'o')
            legend = [f'{i} poles' for i in range(1, NP+3)]+['h']
            plt.legend(legend)
            plt.grid()
            plt.show()


        #print('Poles are:')
        #print(all_results[best_num])

        # If we found more poles than the user requested, we should discard fastest ones
        # BUT we can't discard half of a conjugate pair. If that happens we should look at
        # versions that had higher error
        ps = all_results[best_num]
        num = best_num
        def is_conj_pair(x, y):
            return abs(np.conj(x) - y) < 1e-6
        while len(ps) > NP:
            if is_conj_pair(ps[NP-1], ps[NP]):
                # uh oh, better use a different result
                #print('NOT SPLITTING')
                num -= 1
                ps = all_results[num]
            else:
                ps = ps[:NP]

        zs, dc = self.get_zeros(h_step, dt, all_results[best_num], len(all_results[best_num])-1)
        # we want to convert dc into a scaling factor, but can't until after trimming



        def trim(xs, n):
            if len(xs) < n:
                return np.concatenate((xs, [-float('inf')]*(n-len(xs))))
            else:
                return xs[:n]
        ps_final, zs_final = trim(ps, NP), trim(zs, NZ)
        num_power_0 = np.prod([-z for z in zs_final if z != 0 and np.isfinite(z)])
        den_power_0 = np.prod([-p for p in ps_final if p != 0 and np.isfinite(p)])
        scale = dc / (num_power_0 / den_power_0)
        #print('dc was', dc, 'scale is', scale)
        return ps_final, zs_final, scale

    def extract_pzs(self, h, t, NP, NZ, known_poles=[]):
        h_orig = h
        t_orig = t
        dts = np.diff(t)
        if (max(dts) - min(dts)) / max(dts) > 1e-6:
            h, dt = self.resample(h, t)
        else:
            dt = (t[-1] - t[0]) / (len(t) - 1)

        res = self.extract_pzs_inner(h_orig, t_orig, NP, NZ, known_poles)
        if self.debug:
            self.debug_plot(h_orig, dt, *res, custom_t=t_orig)
        return res

if __name__ == '__main__':
    ma = ModalAnalysis()
    dt = 1e-12
    h_step = ma.get_test_h_step()
    ps, zs, scale = ma.extract_pzs_uniform(h_step, dt, 4, 3, [0, -2e9*dt])
    print()
    print('ps', ps)
    print('zs', zs)
    print('scale', scale)

    ma.debug_plot(h_step, dt, ps, zs, scale)

    #fft = np.fft.fft(h_step)
    #freq = np.fft.fftfreq(len(h_step), dt)
    #plt.semilogy(freq, abs(fft), 'o')
    #plt.show()

    print('Done')
