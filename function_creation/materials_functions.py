import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from bo.utils import *
import uuid
import pickle



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
        self.bounds_setting()
        self.normalize_data()
        self.gp = build_gp_dict(*train_gp(self.input_matrix, jnp.array([self.output_matrix]).T, self.gp_restarts))
        if self.obj_type == "min":
            self.f_opt = -self.output_matrix.min()
        else:
            self.f_opt = self.output_matrix.max()

    def dataset_grouping(self):
        ds_grouped = self.dataset.groupby(self.x_names)[self.y_name].agg(lambda x: x.unique().mean())
        self.dataset = ds_grouped.reset_index()

    def bounds_setting(self):
        self.var_bounds = np.array([self.dataset.iloc[:, :-1].min().values, self.dataset.iloc[:, :-1].max().values]).T
        self.bounds = np.array([[0, 1] for _ in range(self.dim)])

    def normalize_data(self):
        self.input_matrix = (self.dataset.iloc[:, :-1].values - self.var_bounds[:, 0]) / (self.var_bounds[:, 1] - self.var_bounds[:, 0])
        self.output_matrix = (self.dataset[self.y_name].values - np.mean(self.dataset[self.y_name].values)) / np.std(self.dataset[self.y_name].values)

    def __call__(self, x):
        m_y, v_y = inference(self.gp, jnp.array([[x]]))
        if self.obj_type == "min":
            return -m_y.item()
        else:
            return m_y.item()


class Perovskite(GeneralObjective):
    def __init__(self, gp_restarts):
        super().__init__(
            gp_restarts,
            name= "Perovskite",
            expertise = "The creation of environmentally friendly, stable alloyed organic-inorganic perovskites",
            objective_description = '''
            28 spin-coated thin-film samples are examined in situ in parallel using an environmental chamber
            under 85% relative humidity (RH) and 85°C in the air. 
            Sun visible only illumination is applied to enable automatic image capture every 5 min using an RGB camera (~200 μm resolution). 
            Photoactive α-perovskite phases within CsxMAyFA1−x−yPbI3 exhibit a band gap of ~1.5 eV, 
            whereas their main degradation products under hot and humid conditions, 
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


class AutoAM(GeneralObjective):
    def __init__(self,gp_restarts):
        super().__init__(gp_restarts,name = "AutoAM",
        expertise = "Materials exploration and development for three-dimensional (3D) printing technologies.",
        objective_description = '''
        Here, we focused solely on four fundamental syringe extrusion parameters that influence easily distinguished geometric aspects of the leading segment of a printed line. 
        These parameters are ‘prime delay,’ ‘print speed,’ ‘x-position,’ and ‘y-position’
        In each experiment, AM ARES printed a 12 mm line and captured an image of the leading segment. 
        An image analysis module returned a single ‘objective score’ based on the two-dimensional size, shape, and location of the printed feature. 
        Here, the target shape for the leading segment of printed lines was defined as a combined rectangle and semi-circle. 
        To elucidate the effectiveness of the optimization process, we intentionally set the conditions so as to be relatively challenging: 
        We selected a 0.42 mm dispensing tip, and the target shape for the leading segment was almost three times wider at w = 1.2 mm.
        We formulated an objective-scoring algorithm that returned the quotient of the effective specimen area divided by the desired region’s area.
        The effective area is defined as the area of the specimen internal to the desired region less the area of the specimen external to the desired region.
        Negative values for effective area are set to zero. 
        An ideal print, wherein the outline is completely filled without any specimen external to the outline, would achieve a maximal objective score of 1.0.
        The objective is to maximise this score.
        ''',
        obj_type = "max")

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


class P3HT(GeneralObjective):
    def __init__(self, gp_restarts):
        super().__init__(gp_restarts,name = "P3HT",
        expertise = "The Electrical Conductivity of P3HT-CNT Composites",
        objective_description = '''
        In this study, we mix rr-P3HT with four types of Carbon Nanotube Composites (CNTs), where the interactions between the P3HT chains and CNTs are expected to create different morphologies and crystalline structures that control the electrical conductivity of the composite film
        It has already been reported that doping increases the electrical conductivity of all P3HT/CNT composites irrespective of the type of CNTs.
        The workflow begins with data generation from the high-throughput experimental platform, where P3HT/CNT composite films are prepared in a microfluidic reactor linked to an automated drop-casting system, then transitions to rapid optical, and electrical diagnostics.
        In this study, we mix rr-P3HT with four types of CNTs, where the interactions between the P3HT chains and CNTs are expected to create different morphologies and crystalline structures that control the electrical conductivity of the composite film. 
        The types of CNTs used in this study are: 1) long single wall CNTs of lengths in the range of 5–30 µm (l-SWNTs), 2) short single wall CNTs of lengths in the range of 1–3µm, (s-SWNTs), 3) multi walled CNTs (MWCNTs), and 4) double-walled CNTs (DWCNTs). 
        The choice of the nanotubes was aimed to cover a broad range of properties.

        The objective is to maximise the electrical conductivity of the composite inorganic-organic hybrid material film. 
        ''',
        obj_type = "max")





# f = AgNP()
# f = Perovskite()
# f = AutoAM()
# f = CrossedBarrel()
# f = P3HT(gp_restarts = 1)
