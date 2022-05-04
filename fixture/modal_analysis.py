import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

class ModalAnalysis:
    debug = False

    def __init__(self, t, h_step):
        self.scale = t[-1]/2.0
        self.t = t / self.scale
        self.h_step = h_step


    @classmethod
    def get_debug_h_step(cls):
        np.random.seed(4)
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

        #if cls.debug:
        #    plt.figure()
        #    plt.plot(t, h_step, '+-')
        #    plt.grid()
        #    plt.title('Generated debug step response')
        #    plt.show()
        return t, h_step

    def debug_plot(self, ps, NZ):
        zs, dc = self.get_zeros(ps, NZ)
        h_step_est = self.step_response_from_pz(ps, zs, dc)
        print('Plotting ps, zs, dc', ps, zs, dc)
        t = self.t*self.scale

        plt.figure()
        plt.plot(t, self.h_step, '--+')
        plt.plot(t, h_step_est, '-+')
        plt.legend(['Step response', 'Fit'])
        plt.grid()
        plt.show()


    def get_zeros(self, poles, NZ):
        #print('get_zeros with ', poles, dt, NZ)
        assert len(set(poles)) == len(poles), 'get_zeros cannot handle repeated roots'
        NP = len(poles)
        #print(NP, poles)
        # If you leave the timescale as 1, you probably get precision issues
        #timescale_dt = 1/(max(abs(poles)) * len(h_step))
        poles_column = np.array(poles).reshape((len(poles), 1))
        h_step_column = self.h_step.reshape((len(self.h_step), 1))
        exps = np.exp(poles_column * self.t)
        if np.any(np.isinf(exps)):
            # even choosing coefficients of 0 won't fix this
            nans = np.array([float('NaN')]*NZ)
            return nans, float('NaN')

        # Normally, we can find optimal coefficients for each row of exps with linear regression
        # But when we are restricted to fewer zeros, we have a problem.
        # We have to ensure that after doing inverse PFD with our coefficients, the leading term(s) in the numerator go to zero
        # c0/(s-r0) + c1/(s-r1) + c2/(s-r2) -> c0(s-r1)(s-r2) + c1(s-r0)(s-r2) + c2(s-r0)(s-r1) = n2*s^2 + n1*s + n0
        # we can find a square matrix Z~ s.t. N~ = Z~ C, where the ~ means we're doing the large version including required zero entries
        # C = (Z~)^-1 N~
        # Because the bottom entries of N~ are required to be zero, we can drop them and the corresponding right columns from (Z~)^-1
        # C = Zinv N    (3x1) = (3x2) (2x1)
        # Now we use that in our regression to find the coefficients
        # A C = B           (100x3)(3x1) = (100x1)
        # A (Zinv N) = B    (100x3)(3x2)(2x1) = (100x1)
        # N = (A Zinv)^-1 B (Note that we are using a pseudoinverse)
        # And now N is fewer entries, and we can pad the bottom with zeros if we like
        Z_tilde = np.zeros((NP, NP), dtype=poles.dtype)
        poly = np.polynomial.polynomial
        for i in range(NP):
            temp = poly.polyfromroots([poles_column[j][0] for j in range(NP) if j != i])
            for k, c in enumerate(temp):
                Z_tilde[k, i] = c

        # I think this should be invertible as long as there are no repeated roots (which is not allowed)
        Z_tilde_inv = np.linalg.inv(Z_tilde)
        Z_inv = Z_tilde_inv[:, :NZ+1]

        #print(exps.shape, Z_inv.shape, h_step.shape)
        # N is the transfer function numerator, with constant term first
        N = np.linalg.pinv((exps.T @ Z_inv)) @ h_step_column
        zeros = np.roots(N.reshape(NZ+1)[::-1])

        #if self.debug:
        #    h_step_est = exps.T @ (Z_inv @ N)
        #    #h_step_est_2 = exps.T @ coefs
        #    plt.figure()
        #    plt.plot(t, h_step, 'o')
        #    plt.plot(t, np.real(h_step_est))
        #    plt.legend(['Orig', 'est'])
        #    plt.grid()
        #    plt.title('Using numerator coefficients instead of zeros')
        #    plt.show()


        # when we take the roots of N we lose scale information
        #scale = N[-1]
        # rather than store scale we relate it to a circuit parameter, DC
        dc = N[0] / np.prod([-p for p in poles if p != 0])
        zeros_sorted = zeros[np.lexsort((abs(zeros),))]
        return zeros_sorted, dc

    def step_response_from_pz(self, ps, zs, dc):
        zs = zs[np.isfinite(zs)]
        ps = ps[np.isfinite(ps)]

        from scipy.signal import residue
        r, p, k = residue(np.poly(zs), np.poly(ps))

        scale = self.get_scale(ps, zs, dc)
        h_step_est = np.zeros(self.h_step.shape)
        for coef, pole in zip(r, p):
            h_step_est = h_step_est + scale * coef * np.exp(pole*self.t)
            #print('Added coef', scale*coef, 'for pole', pole)
        return h_step_est

    def error_from_poles(self, poles, NZ):
        zs, dc = self.get_zeros(poles, NZ)
        h_step_est = self.step_response_from_pz(poles, zs, dc)
        error = np.sum((h_step_est - self.h_step)**2, 0)
        return error

    def get_scale(self, ps, zs, dc):
        # by default our rational polynomial has the high-order term 1,
        # because that lets us represent poles/zeros at 0
        # The extra scale term out front can be found from the DC value
        num_power_0 = np.prod([-z for z in zs if z != 0 and np.isfinite(z)])
        den_power_0 = np.prod([-p for p in ps if p != 0 and np.isfinite(p)])
        scale = dc / (num_power_0 / den_power_0)
        return scale

    def poles_from_coefs(self, coefs):
        return np.roots(np.concatenate(([1], coefs)))

    def coefs_from_poles(self, poles):
        # np.poly puts the constant term last, and always returns a leading 1
        return np.poly(poles)[1:]

    def _fit_poles(self, NP, NZ, known_poles):
        def err_minimizer(coefs):
            poles = np.concatenate((known_poles, self.poles_from_coefs(coefs)))
            err = self.error_from_poles(poles, NZ)
            if np.isnan(err):
                # if we return nan, minimizer will return nan
                return float('inf')
            return err

        max_freq_guess = 2.0/self.t[-1]
        num_poles_guess = NP - len(known_poles)
        poles_guess = np.linspace(-max_freq_guess/num_poles_guess,
                                  -max_freq_guess,
                                  num_poles_guess)

        #poles_guess = np.array([-2e9, -20e9])*self.scale

        if self.debug:
            print('guessing ps', poles_guess)
            self.debug_plot(np.concatenate((known_poles, poles_guess)), NZ)
        coefs_guess = self.coefs_from_poles(poles_guess)
        minimizer = scipy.optimize.minimize(err_minimizer, coefs_guess)
        coefs_opt = minimizer.x
        poles_opt = np.concatenate((known_poles, self.poles_from_coefs(coefs_opt)))

        if self.debug:

            print('got coefs', coefs_opt, 'poles', poles_opt)
            self.debug_plot(poles_opt, NZ)

        zeros_opt, dc_opt = self.get_zeros(poles_opt, NZ)

        return poles_opt, zeros_opt, dc_opt


    def extract_pzs(self, NP, NZ, known_poles):
        # TODO scaling
        ps_scaled, zs_scaled, dc = self._fit_poles(NP, NZ, known_poles)
        ps = ps_scaled / self.scale
        zs = zs_scaled / self.scale

        # TODO this scale is unrelated to the scale above
        scale = self.get_scale(ps, zs, dc)
        return ps, zs, scale



if __name__ == '__main__':
    #dt = 1e-12
    t, h_step = ModalAnalysis.get_debug_h_step()
    ma = ModalAnalysis(t, h_step)
    ps, zs, scale = ma.extract_pzs(4, 1, [0])
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


