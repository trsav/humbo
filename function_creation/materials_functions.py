import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from bo.utils import *
import uuid
import pickle

jax.config.update("jax_enable_x64", True)

class GeneralObjective:
    def __init__(self, gp_restarts,name, expertise, objective_description, obj_type):
        self.expertise = expertise
        self.objective_description = objective_description
        self.obj_type = obj_type
        self.name = name
        dataset_path = "materials_benchmarking/datasets/"+self.name+"_dataset.csv"
        self.dataset = pd.read_csv(dataset_path)
        self.x_names = list(self.dataset.columns)[:-1]
        self.y_name = list(self.dataset.columns)[-1]
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


class Perovskite(GeneralObjective):
    def __init__(self, gp_restarts):
        super().__init__(
            gp_restarts,
            name= "Perovskite",
            expertise = "The creation of environmentally friendly, stable alloyed organic-inorganic perovskites",
            objective_description = '''
            28 spin-coated thin-film samples are examined in situ in parallel using an environmental chamber under 85% relative humidity (RH) and 85°C in the air. 
            Sun visible only illumination is applied to enable automatic image capture every 5 min using an RGB camera (~200 μm resolution). 
            Photoactive α-perovskite phases within CsxMAyFA1−x−yPbI3 exhibit a band gap of ~1.5 eV, whereas their main degradation products under hot and humid conditions, 
            PbI2 (2.27 eV)25 δ-CsPbI3 (2.82 eV),26 or δ-FAPbI3 (2.43 eV)27 show deteriorated photophysical properties. 
            We hence used a color-based metric as a proxy to capture the macroscopic evolution of the high-band-gap, 
            non-perovskite phases. We define the instability index as the integrated color change of an unencapsulated perovskite film over accelerated degradation test duration T. 
            the instability index is a function of time, area-averaged, color-calibrated red, green, and blue pixel values of the sample. 
            The cutoff time was set to T = 7,000 min based on the observed divergence between the most- and least-stable compositions. 

            The objective is the negative of the instability index, which must be maximised.
            ''',
            obj_type = "min"
        )

class AgNP(GeneralObjective):
    def __init__(self, gp_restarts):
        super().__init__(gp_restarts,name = "AgNP",
        expertise = "Microfluidic high-throughput experimental (HTE) for silver nanomaterial synthesis.",
        objective_description = '''
        Nanoparticles are synthesized in aqueous sub-microliter droplets. 
        In such a flow system, the concentration of each reactant is directly proportional to the flow rate ratio Qi (%) between the flow rate of the reactant and the total aqueous flow rate. 
        By adjusting the flow rate of the solvent (water), the flow rate ratios Qseed of silver seeds, 𝑄AgNO3 of silver nitrate, QTSC of trisodium citrate and QPVA of PVA are independently controlled by varying the flow rate of the corresponding solutions using LabView automated syringe pumps. 
        The flow rate ratio QAA of ascorbic acid is kept constant. 
        The mixing of the reactants inside the droplet depends on the speed of the droplet, which is directly proportional to the total flow rate Qtotal (µL/min) of both oil and aqueous phases. 
        The absorbance spectra of the droplets are measured inline, and the five controlled variables Qseed, 𝑄AgNO3, QTSC, QPVA and Qtotal are used as input parameters for the two-step optimization framework and the absorbance spectra as the output.
        The loss (or objective) applies cosine similarity between the measured and targeted spectra, quantifying the shape similarity of the two spectra.

        The objective is the negative of this loss value, which must be maximised.
        ''',
        obj_type = "min")


class CrossedBarrel(GeneralObjective):
    def __init__(self,gp_restarts):
        super().__init__(gp_restarts,name = "Crossed barrel",
        expertise = "The mechanical testing system of (3D) printing to determine the mechanical properties such as toughness of crossed-barrel structures.",
        objective_description = '''
        Toughness is difficult to optimize because it requires maximizing a combination of two properties that tend to be inversely correlated, namely, strength and ductility.
        Defined as the area under the force (F)–displacement (D) curve, toughness represents how much energy a component can absorb before failure, 
        which makes it an important property to optimize in the context of design for safety and failure tolerance.

        The general shape optimised is a crossed barrel, consisting of two flat disks, between which are a number of struts. 
        Parameter information: n denotes the number of struts between the two disks that are being compressed, 
        t is the thickness of each strut, r is the inner radius of each strut.
        theta denotes a twist of the struts, enabling them to cross over.
        Each given design is 3D printed and experimentally tested.

        The objective is to maximise the toughness of a design, there will be a trade off between weight and strength.
        ''',
        obj_type = "max")
