import numpy as np


class KalmanFilter():
    def __init__(self, dt, acc_std=20):
        """
        Args:
            dt: prediction time interval;
            meas_std: measure standard deviation, it describes the uncertainty of 
                measurement;
            acc_std: accerlation standard deviation, it describes 
                the uncertainty of motion prediction;

        distribution:
            bel_bar(xt) ~ N(xt_bar, Σt_bar)
            bel(xt) ~ N(xt, Σt)
        predict: 
            Xt = A * Xt-1 + B*ut + εt
            bel(xt-1) -> bel_bar(xt)
            var(εt) = R
        correct: 
            Zt = C * Xt + δt
            bel(xt-1)_bar -> bel(xt)
            var(δt) = Q
        
        """
        self.dt = dt

        # bel distribution(initial state)
        self.x = np.array([[0], 
                           [0],
                           [0],
                           [0]], dtype=np.float32)
        self.sigma = np.eye(self.x.shape[0], dtype=np.float32) * 500

        # bel bar distribution
        self.x_bar = self.x.copy()
        self.sigma_bar = self.sigma.copy()


        # predict
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.B = np.array([[-0.5*dt**2, 0],
                           [0, -0.5*dt**2],
                           [-dt, 0],
                           [0, -dt]], dtype=np.float32)
        self.R = np.array([[0.25*dt**4, 0, 0.5*dt**3, 0],
                           [0, 0.25*dt**4, 0, 0.5*dt**3],
                           [0.5*dt**3, 0, dt**2, 0],
                           [0, 0.5*dt**3, 0, dt**2]], dtype=np.float32) \
                            * acc_std**2

        # correct
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.base_Q = np.array([[1, 0],
                                [0, 1]], dtype=np.float32)
        
    def init(self, x_init):
        if isinstance(x_init, list) or isinstance(x_init, tuple):
            x_init = np.array(x_init)
        if x_init.shape != (4, 1):
            x_init = x_init.reshape(4, 1)
        self.x = x_init
        self.x_bar = x_init
        # call correct to shrink self.sigma
        self.correct(x_init[:2, :], meas_std=0.01)

        
    def predict(self, ut):
        # u_(t) = [[ux_(t)], [uy_(t)]] or [ux_(t), uy_(t)]
        if isinstance(ut, list) or isinstance(ut, tuple):
            ut = np.array(ut)
        if ut.shape != (2, 1):
            ut = ut.reshape(2, 1)
        # Prediction mean: x_(t)_bar = A*x_(t-1) + B*u_(t)   
        self.x_bar = self.A @ self.x + self.B @ ut

        # Prediction covariance: Σ_bar = A*Σ*A' + R         
        self.sigma_bar = self.A @ self.sigma @ self.A.T + self.R
        return self.x_bar.flatten()[0:2]  # only return x, y 

    def correct(self, zt, meas_std=5):
        # z_(t) = [[rel_x], [rel_y]] or [rel_x, rel_y]
        if zt is None:  # no measurement 
            self.x = self.x_bar
            self.sigma = self.sigma_bar
            return self.x.flatten()[:2]

        if isinstance(zt, list) or isinstance(zt, tuple):
            zt = np.array(zt)
        
        if zt.shape != (2, 1):
            zt = zt.reshape(2, 1)
        Q = self.base_Q * meas_std**2
        # term = C*Σ_bar*C'+ Q
        term = self.C @ self.sigma_bar @ self.C.T + Q

        # Kalman Gain
        # K = Σ_bar*C'*inv(C*Σ_bar*C'+ Q)
        # K = np.dot(np.dot(self.sigma_bar, self.C.T), np.linalg.inv(term))  
        K = self.sigma_bar @ self.C.T @ np.linalg.inv(term)
        
        # x = x_bar + K*(z - C*x_bar)  
        self.x = self.x_bar + K @ (zt - self.C @ self.x_bar)

        # Update Σ
        # ∑ = (I - K*C)∑_bar
        I = np.eye(self.C.shape[1])
        # self.sigma = np.dot((I - np.dot(K, self.C)), self.sigma_bar)   
        self.sigma = (I - K @ self.C) @ self.sigma_bar

        return self.x.flatten()[:2]
    
    # def get_position(self):
    #     return self.x.flatten()[:2]

    # def get_velocity(self):
    #     return self.x.flatten()[2:4]
    
    # def get_predcited_position(self):
    #     return self.x_bar.flatten()[:2]
    
    # def get_predcited_velocity(self):
    #     return self.x_bar.flatten()[2:4]

    def get_x(self):
        return self.x

    def get_x_bar(self):
        return self.x_bar
    
    def get_sigma(self):
        return self.sigma
    
    def get_sigma_bar(self):
        return self.sigma_bar
    


class KalmanFilterNoNumpy():
    "no numpy needed"
    def __init__(self, dt, acc_std=20):
        self.dt = dt

        # bel distribution(initial state)
        # state x
        self.x = 0.0
        self.y = 0.0 
        self.vx = 0.0
        self.vy = 0.0
        # sigma
        self.var_x = 500
        self.var_y = 500
        self.var_vx = 500
        self.var_vy = 500
        self.cov_x_vx = 0
        self.cov_y_vy = 0

        # bel_bar distribution(initial state)
        # state x_bar
        self.x_bar = 0.0
        self.y_bar = 0.0 
        self.vx_bar = 0.0
        self.vy_bar = 0.0
        # sigma
        self.var_x_bar = 500
        self.var_y_bar = 500
        self.var_vx_bar = 500
        self.var_vy_bar = 500
        self.cov_x_vx_bar = 0
        self.cov_y_vy_bar = 0

        self.acc_std = acc_std

    def init(self, x_init):
        if isinstance(x_init, np.ndarray):
            x_init = x_init.flatten().tolist()
        self.x = x_init[0]
        self.y = x_init[1]
        self.vx = x_init[2]
        self.vy = x_init[3]
        self.x_bar = x_init[0]
        self.y_bar = x_init[1]
        self.vx_bar = x_init[2]
        self.vy_bar = x_init[3]

        # call correct to shrink self.sigma
        self.correct((x_init[0], x_init[1]), meas_std=0.01)

    def predict(self, ut):
        # u_(t) = [ux_(t), uy_(t)]
        if isinstance(ut, np.ndarray):
            ut = ut.flatten().tolist()

        uxt, uyt = ut
        # update bel_bar distribution: mean
        self.x_bar = self.x + self.vx * self.dt - 0.5 * (self.dt)**2 * uxt
        self.y_bar = self.y + self.vy * self.dt - 0.5 * (self.dt)**2 * uyt
        self.vx_bar = self.vx - self.dt * uxt
        self.vy_bar = self.vy - self.dt * uyt

        # update bel_bar distribution: variance
        m = (self.acc_std)**2
        self.var_x_bar = self.var_x + 2 * self.dt * self.cov_x_vx \
            + (self.dt)**2 * self.var_vx + 0.25 * (self.dt)**4 * m
        self.var_y_bar = self.var_y + 2 * self.dt * self.cov_y_vy \
            + (self.dt)**2 * self.var_vy + 0.25 * (self.dt)**4 * m
        self.var_vx_bar = self.var_vx + (self.dt)**2 * m
        self.var_vy_bar = self.var_vy + (self.dt)**2 * m
        self.cov_x_vx_bar = self.cov_x_vx + self.dt * self.var_vx \
            + 0.5 * (self.dt)**3 * m
        self.cov_y_vy_bar = self.cov_y_vy + self.dt * self.var_vy \
            + 0.5 * (self.dt)**3 * m
        
        return self.x_bar, self.y_bar  # only return x, y 
    
    def correct(self, zt, meas_std=5):
        # z_(t) = [zx_(t), zy_(t)]
        if zt is None:  # no measurement 
            self.x = self.x_bar  
            self.y = self.y_bar
            self.vx = self.vx_bar
            self.vy = self.vy_bar
            self.var_x = self.var_x_bar
            self.var_y = self.var_y_bar
            self.var_vx = self.var_vx_bar
            self.var_vy = self.var_vy_bar
            self.cov_x_vx = self.cov_x_vx_bar
            self.cov_y_vy = self.cov_y_vy_bar
            return self.x, self.y

        if isinstance(zt, np.ndarray):
            zt = zt.flatten().tolist()

        zxt, zyt = zt
        n = meas_std**2
        term_1 = 1 / (self.var_x_bar + n)
        term_2 = 1 / (self.var_y_bar + n)

        # update bel distribution: mean
        self.x = self.x_bar + self.var_x_bar * (zxt - self.x_bar) * term_1
        self.y = self.y_bar + self.var_y_bar * (zyt - self.y_bar) * term_2
        self.vx = self.vx_bar + self.cov_x_vx_bar * (zxt - self.x_bar) * term_1
        self.vy = self.vy_bar + self.cov_y_vy_bar * (zyt - self.y_bar) * term_2

        # update bel distribution: variance
        self.var_x = self.var_x_bar - self.var_x_bar**2 * term_1
        self.var_y = self.var_y_bar - self.var_y_bar**2 * term_2
        self.var_vx = self.var_vx_bar - self.cov_x_vx_bar**2 * term_1
        self.var_vy = self.var_vy_bar - self.cov_y_vy_bar**2 * term_2
        self.cov_x_vx = self.cov_x_vx_bar - self.var_x_bar * self.cov_x_vx_bar * term_1
        self.cov_y_vy = self.cov_y_vy_bar - self.var_y_bar * self.cov_y_vy_bar * term_2

        return self.x, self.y

    def get_x(self):
        x = np.array([[self.x],
                      [self.y],
                      [self.vx],
                      [self.vy]], dtype=np.float32)
        return x

    def get_x_bar(self):
        x_bar = np.array([[self.x_bar], 
                          [self.y_bar],
                          [self.vx_bar], 
                          [self.vy_bar]], dtype=np.float32)
        return x_bar
    
    def get_sigma(self):
        sigma = np.array([[self.var_x, 0, self.cov_x_vx, 0],
                          [0, self.var_y, 0, self.cov_y_vy],
                          [self.cov_x_vx, 0, self.var_vx, 0],
                          [0, self.cov_y_vy, 0, self.var_vy]], dtype=np.float32)
        return sigma
    
    def get_sigma_bar(self):
        sigma_bar = np.array([[self.var_x_bar, 0, self.cov_x_vx_bar, 0],
                          [0, self.var_y_bar, 0, self.cov_y_vy_bar],
                          [self.cov_x_vx_bar, 0, self.var_vx_bar, 0],
                          [0, self.cov_y_vy_bar, 0, self.var_vy_bar]], dtype=np.float32)
        return sigma_bar


        


    