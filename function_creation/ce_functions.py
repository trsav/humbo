import sys
from matplotlib import gridspec
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from bo.utils import *
import uuid
import pickle
import scipy.integrate as integrate

jax.config.update("jax_enable_x64", True)

class GeneralObjective:
    def __init__(self, gp_restarts,name, expertise, objective_description, obj_type):
        self.expertise = expertise
        self.objective_description = objective_description
        self.obj_type = obj_type
        self.name = name
        dataset_path = "function_creation/datasets/"+self.name+"_dataset.csv"
        self.dataset = pd.read_csv(dataset_path)
        self.x_names = list(self.dataset.columns)[:-1]
        self.y_name = list(self.dataset.columns)[-1]
        self.y_range = self.dataset[self.y_name].max() - self.dataset[self.y_name].min()
        self.dim = len(self.x_names)
        self.gp_restarts = gp_restarts
        self.dataset_grouping()
        self.normalize_data()
        self.bounds_setting()

        self.gp = build_gp_dict(*train_gp(self.input_matrix, jnp.array([self.output_matrix]).T, self.gp_restarts))

    def dataset_grouping(self):
        ds_grouped = self.dataset.groupby(self.x_names)[self.y_name].agg(lambda x: x.unique().mean())
        self.dataset = ds_grouped.reset_index()

    def bounds_setting(self):
        self.bounds = np.array([self.dataset.iloc[:, :-1].min().values, self.dataset.iloc[:, :-1].max().values]).T

    def normalize_data(self):
        self.input_matrix = self.dataset.iloc[:, :-1].values 
        self.output_matrix = self.dataset[self.y_name].values
        if self.obj_type == "min":
            self.f_opt = -self.output_matrix.min()
        else:
            self.f_opt = self.output_matrix.max()
        self.input_mean = self.input_matrix.mean(axis=0)
        self.input_std = self.input_matrix.std(axis=0)
        self.output_mean = self.output_matrix.mean()
        self.output_std = self.output_matrix.std()
        self.input_matrix = (self.input_matrix - self.input_mean) / self.input_std
        self.output_matrix = (self.output_matrix - self.output_mean) / self.output_std

    def __call__(self, x):
        x = np.array(x)
        x_n = [float((x[i] - self.input_mean[i]) / self.input_std[i]) for i in range(len(x))]
        m_y, v_y = inference(self.gp, jnp.array([x_n]))
        val = (m_y.item() * self.output_std) + self.output_mean
        if self.obj_type == "min":
            return -val
        else:
            return val


class SelfOpt(GeneralObjective):
    def __init__(self, gp_restarts):
        super().__init__(
            gp_restarts,
            name= "selfopt",
        expertise = "Optimization in chemical reaction conditions.",
        objective_description = '''
        The system represents an exothermic continuous-flow reactor, where the primary focus is on maximizing the final concentration of the desired product while ensuring safety and operational stability. 
        The reactor is characterized by two main input variables: the reactor temperature \( T_w \) and the Damköhler number \( Da \).
        
        \( T_w \) controls the kinetics of the reaction, influencing both the rate and the equilibrium conversion. 
        \( Da \) is a dimensionless number that balances reaction and diffusion rates, effectively affecting how uniformly the reaction proceeds throughout the reactor.

        The objective function aims to maximize the final product concentration.
        ''',
        obj_type = "max")


class Reactor(GeneralObjective):
    def __init__(self, gp_restarts):
        super().__init__(
            gp_restarts,
            name= "reactor",
        expertise = "Pulsed-flow helical tube reactor design and fluid dynamics.",
        objective_description = '''
    A helical-tube reactor is parameterized by a coil radius, coil pitch, and inversion. 
    Coil pitch denoted by controls how extended the helical tube is, coil radius denoted by controls how tight the coils are within the helical tube, 
    and the inversion parameter is denoted by  controls the change in coil direction and specifies where along the coil the inversion takes place. 
    The length of the coil is maintained as fixed, resulting in all parameterized coils having the same volume. 
    Within the parameterization, we include a fixed-length inlet and outlet to the coil. 
    The inlet and outlet are horizontal, and a smooth interpolation is used to ensure that the transition from inlet to coil and coil to outlet, is smooth. 

    The reactor inlet flow is at a Reynolds number of 50 for which relatively insignificant mixing is expected to take place. 
    A superimposition of oscillatory velocity is, therefore, needed to operate under a wide range of plug flow conditions. 
    This oscillatory velocity is achieved through parameters representing oscillation amplitude and frequency which will effect the vortices and fluid behaviour. 

    The objective function is to maximise the equivalent number of tanks-in-series for a given reactor. A higher value represents 
    a closer approximation to plug-flow behaviour, and better mixing. 
        ''',
        obj_type = "max")

    # create a method which produces a plot of a given solution (x)
    @staticmethod
    def plot_solution(x,path_name):

        a, f, re, pitch, coil_rad = x
        v = (re * 9.9 * 10**-4) / (990 * 0.005)

        fig = plt.figure(figsize=(9, 3))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5]) 
        ax = fig.add_subplot(gs[0])
        x = np.linspace(0, 1, 300)
        y = np.sin(2 * np.pi * f * x) * a + v
        ax.plot(x, y,color='black',lw=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Inlet Velocity (m/s)')
        ax.set_xlim([0,1])
        ax.set_ylim([-0.01,0.02])
        ax.set_title('Operating Conditions')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        
        n = 100
        l = 0.0875
        rot = l/(2 * np.pi * coil_rad)
        h = rot * pitch 

        rho = np.linspace(0, 2 * np.pi * rot, n)    
        z = np.linspace(0, h, n)
        theta = np.array([coil_rad for i in range(n)])

        x = theta * np.cos(rho)
        y = theta * np.sin(rho)
        z = z
        ax.grid(alpha=0.5)

        ax = fig.add_subplot(gs[1])
        ax.set_aspect('equal')
        ax.set_xlim([-0.02,0.02])
        ax.set_ylim([0,0.02])

        #turn off everything that isn\t needed like ticks etc...
        # ax.set_yticks([])
        ax.set_ylabel('h')
        ax.set_xlabel('x')
        
        ax.set_yticks([0,0.005,0.01,0.015,0.02])
        ax.set_xticks([0.02,0.01,0.0,-0.01,-0.02])
        ax.grid(alpha=0.5)

        # remove grey background 
        ax.set_title('Reactor Geometry')
    
        ax.plot(x,z,lw=3,alpha=0.8,color='black')
        ax.plot(x,z,lw=20,alpha=0.3,color='black')
        fig.tight_layout()
        # fig.subplots_adjust(wspace=0.01,top=0.9,bottom=0.3)
        # fig.show()
        fig.savefig(path_name)


class BioProcess:
    def __init__(self):
        self.expertise = "The optimization of fed-batch bioprocesses."
        self.objective_description = 'NONE'
        self.obj_type = 'max'
        self.name = 'Bioprocesses'

        self.x_names = ['Light Intensity','Nitrogen Feed Rate']
        self.y_name = 'Final Product Concentration'

        self.dim = len(self.x_names)
        self.x0 = [0.0,1.0,150]
        self.tf = 200 

    def dxdt(self,t, y, I,Fn):
        cq,cx,cn = y
        um = 0.0572
        ud = 0.0
        kn = 393.1
        ynx = 504.1
        km = 0.00016
        kd = 0.281
        ks = 178.9
        ki = 447.1
        ksq = 23.51
        kiq = 800
        knp = 16.89 

        dcq = km * ((I)/(I+ksq+((I**2)/kiq)))*cx*((cn)/(cn+kn))-kd*((cq)/(cn+knp))
        dcx = um * ((I)/(I+ks+((I**2)/ki)))*cx*((cn)/(cn+kn))-ud*cx
        if cn <= 500 and cx > 10:
            dcn = -ynx*um * ((I)/(I+ksq+((I**2)/kiq)))*cx*((cn)/(cn+kn)) + Fn
        else:
            dcn = -ynx*um * ((I)/(I+ksq+((I**2)/kiq)))*cx*((cn)/(cn+kn)) + Fn
        return [dcq,dcx,dcn]

    def __call__(self, x):

        sol = integrate.solve_ivp(self.dxdt, [0, self.tf], self.x0, args=([x[0],x[1]]), method='RK45', rtol=1e-6, atol=1e-8)

        return sol.y[0][-1]
        
    @staticmethod
    def plot_solution(x,path_name):
        plt.figure()
        fig,ax = plt.subplots(1,2,figsize=(6,3))
        labels = ['Light Intensity','Nitrogen Feed Rate']
        lims = [[0,400],[0,40]]

        for i in range(2):
            ax[i].bar([0],[x[i]],color='k')
            ax[i].set_xticks([0])
            ax[i].set_xticklabels([labels[i]])
            ax[i].set_ylim(lims[i])

        plt.tight_layout()
        plt.show()

        plt.savefig(path_name)

# f = BioProcess()

class BioProcess_Profile:
    def __init__(self,n_control):
        self.expertise = "The optimization of fed-batch bioprocesses."
        self.objective_description = 'NONE'
        self.obj_type = 'max'
        self.f_opt = 0 
        self.name = 'bioprocess_profile'
        self.tf = 200 
        self.n_control = n_control
        x_names = []

        t_step = self.tf/n_control

        for i in range(n_control):
            x_names.append('I('+str(int(i*t_step))+'-'+str(int((i+1)*t_step))+')')
        for i in range(n_control):
            x_names.append('Fn('+str(int(i*t_step))+'-'+str(int((i+1)*t_step))+')')
        
        self.x_names = x_names

        # self.x_names = ['I(0-50)','I(50-100)','I(100-150)','I(150-200)','Fn(0-50)','Fn(50-100)','Fn(100-150)','Fn(150-200)']

        self.y_name = 'Final Product Concentration'

        self.dim = len(self.x_names)
        self.x0 = [0.0,1.0,150]
        self.y_range = 0.03
        self.dim = len(self.x_names)\
        
        self.bounds = np.array([[0,400] for i in range(n_control)]+[[0,40] for i in range(n_control)])

    def dxdt(self,t, y, I,Fn):
        cq,cx,cn = y
        um = 0.0572
        ud = 0.0
        kn = 393.1
        ynx = 504.1
        km = 0.00016
        kd = 0.281
        ks = 178.9
        ki = 447.1
        ksq = 23.51
        kiq = 800
        knp = 16.89 

        dcq = km * ((I)/(I+ksq+((I**2)/kiq)))*cx*((cn)/(cn+kn))-kd*((cq)/(cn+knp))
        dcx = um * ((I)/(I+ks+((I**2)/ki)))*cx*((cn)/(cn+kn))-ud*cx
        if cn <= 500 and cx > 10:
            dcn = -ynx*um * ((I)/(I+ksq+((I**2)/kiq)))*cx*((cn)/(cn+kn)) + Fn
        else:
            dcn = -ynx*um * ((I)/(I+ksq+((I**2)/kiq)))*cx*((cn)/(cn+kn)) + Fn
        return [dcq,dcx,dcn]

    def __call__(self, x):
        
        t_n = int(len(x)/2)
        x0 = self.x0 
        t_step = self.tf/t_n
        for i in range(t_n):
            sol = integrate.solve_ivp(self.dxdt, [0, t_step], x0, args=([x[i],x[i+self.n_control]]), method='RK45', rtol=1e-6, atol=1e-8)

            x0 = sol.y[:,-1]

        return sol.y[0][-1]
        
    @staticmethod
    def plot_result(x,path_name,self):
        plt.figure()
        fig,ax = plt.subplots(1,5,figsize=(12,2.5))
        t_n = int(len(x)/2)
        t_step = self.tf/t_n
        x0 = self.x0 
        total_sol = []
        t = []
        I_store = []
        FCn_store = []

        for i in range(t_n):
            sol = integrate.solve_ivp(self.dxdt, [0,t_step], x0, args=([x[i],x[i+self.n_control]]), method='RK45', rtol=1e-6, atol=1e-8)
            x0 = sol.y[:,-1]
            total_sol.append(sol.y)
            t.append(sol.t)
            I_store.append([x[i] for j in range(len(sol.t))])
            FCn_store.append([x[i+self.n_control] for j in range(len(sol.t))])

        total_sol_appended = total_sol[0]
        t_appended = t[0]
        I_store_appended = I_store[0]
        FCn_store_appended = FCn_store[0]
        for sol in total_sol[1:]:
            total_sol_appended = np.concatenate((total_sol_appended,sol),axis=1)
        for i in range(1,t_n):
            t_appended = np.append(t_appended,t[i]+(i)*t_step)
            I_store_appended = np.append(I_store_appended,I_store[i])
            FCn_store_appended = np.append(FCn_store_appended,FCn_store[i])
        total_sol_appended = total_sol_appended.T

        ax[0].plot(t_appended,total_sol_appended[:,0],c='k')
        ax[0].set_ylabel('Product Concentration')
        ax[1].plot(t_appended,total_sol_appended[:,1],c='k')
        ax[1].set_ylabel('Biomass Concentration')
        ax[2].plot(t_appended,total_sol_appended[:,2],c='k')
        ax[2].set_ylabel('Nitrogen Concentration')
        for ax_ in ax:
            ax_.set_xlabel('Time (hr)')
        ax[3].plot(t_appended,I_store_appended,c='k')
        ax[3].fill_between(t_appended,0,I_store_appended,color='k',alpha=0.5)

        ax[3].set_ylabel('Light Intensity')
        ax[4].plot(t_appended,FCn_store_appended,c='k')
        ax[4].fill_between(t_appended,0,FCn_store_appended,color='k',alpha=0.5)
        ax[4].set_ylabel('Nitrogen Feed Rate')

        ax[0].set_ylim([0,0.05])
        ax[1].set_ylim([0,15])
        ax[2].set_ylim([0,2000])
        ax[3].set_ylim([0,450])
        ax[4].set_ylim([0,45])

        plt.tight_layout()

        print('Plotting result at ',path_name)
        plt.savefig(path_name)
        # save but with name 'most_recent.png'
        path_ = path_name.split('/')[:-1]
        path = '/'.join(path_)
        plt.savefig(os.path.join(path, 'most_recent.png'))
        plt.close()

    @staticmethod
    def plot_solution(x,path_name,self):
        plt.figure()
        fig,ax = plt.subplots(1,2,figsize=(5,2.5))
        t_n = int(len(x)/2)
        t_step = self.tf/t_n
        x0 = self.x0 
        total_sol = []
        t = []
        I_store = []
        FCn_store = []

        for i in range(t_n):
            sol = integrate.solve_ivp(self.dxdt, [0,t_step], x0, args=([x[i],x[i+self.n_control]]), method='RK45', rtol=1e-6, atol=1e-8)
            x0 = sol.y[:,-1]
            total_sol.append(sol.y)
            t.append(sol.t)
            I_store.append([x[i] for j in range(len(sol.t))])
            FCn_store.append([x[i+self.n_control] for j in range(len(sol.t))])

        total_sol_appended = total_sol[0]
        t_appended = t[0]
        I_store_appended = I_store[0]
        FCn_store_appended = FCn_store[0]
        for sol in total_sol[1:]:
            total_sol_appended = np.concatenate((total_sol_appended,sol),axis=1)
        for i in range(1,t_n):
            t_appended = np.append(t_appended,t[i]+(i)*t_step)
            I_store_appended = np.append(I_store_appended,I_store[i])
            FCn_store_appended = np.append(FCn_store_appended,FCn_store[i])
        total_sol_appended = total_sol_appended.T

        ax[0].plot(t_appended,I_store_appended,c='k')
        ax[1].plot(t_appended,FCn_store_appended,c='k')

        ax[0].set_xlabel('Time (hr)')
        ax[0].set_ylabel('Light Intensity')
        ax[1].set_xlabel('Time (hr)')
        ax[1].set_ylabel('Nitrogen Feed Rate')

        ax[0].set_ylim([0,450])
        ax[1].set_ylim([0,45])

        plt.tight_layout()

        plt.savefig(path_name)
        plt.close()


# f = Reactor(1)
# f.plot_solution([0.006,5,30,0.01,0.01],'reac_example.pdf')

n_control = 8
f = BioProcess_Profile(n_control)
x = np.linspace(400,0,n_control)
x = np.append(x,np.linspace(40,0,n_control))
# f.plot_solution(x,'test.pdf',f)
f.plot_result(x,'bioproc_example.pdf',f)