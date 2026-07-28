import pyvista as pv
import os
from tqdm import tqdm

general_path = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\Gen_Metrics_Exp_Files_Ablation"
template_tagged = pv.read(r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\template_tagged_correct.vtk")

path_tosave = r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\Gen_Metrics_Exp_Files_Ablation"


models = ['cnf_model_ablation_activation_elu_0',
             'cnf_model_ablation_activation_elu_1',
             'cnf_model_ablation_activation_elu_2',
             'cnf_model_ablation_activation_leaky_relu_0',
             'cnf_model_ablation_activation_leaky_relu_1',
             'cnf_model_ablation_activation_leaky_relu_2',
             'cnf_model_ablation_activation_relu_0',
             'cnf_model_ablation_activation_relu_1',
             'cnf_model_ablation_activation_relu_2',
             'cnf_model_ablation_activation_silu_0',
             'cnf_model_ablation_activation_silu_1',
             'cnf_model_ablation_activation_silu_2',
              'cnf_model_ablation_n_flow_5_0',
              'cnf_model_ablation_n_flow_5_1',
              'cnf_model_ablation_n_flow_5_2',
              'cnf_model_ablation_n_flow_10_0',
              'cnf_model_ablation_n_flow_10_1',
              'cnf_model_ablation_n_flow_10_2',
              'cnf_model_ablation_n_flow_20_0',
              'cnf_model_ablation_n_flow_20_1',
              'cnf_model_ablation_n_flow_20_2',
              'cnf_model_ablation_n_flow_25_0',
              'cnf_model_ablation_n_flow_25_1',
              'cnf_model_ablation_n_flow_25_2',
              'cnf_model_ablation_n_flow_30_0',
              'cnf_model_ablation_n_flow_30_1',
              'cnf_model_ablation_n_flow_30_2',
               'cnf_model_ablation_out_dim_conf_6_0',
               'cnf_model_ablation_out_dim_conf_6_1',
               'cnf_model_ablation_out_dim_conf_6_2',
               'cnf_model_ablation_out_dim_conf_9_0',
               'cnf_model_ablation_out_dim_conf_9_1',
               'cnf_model_ablation_out_dim_conf_9_2',
               'cnf_model_ablation_out_dim_conf_15_0',
               'cnf_model_ablation_out_dim_conf_15_1',
               'cnf_model_ablation_out_dim_conf_15_2',
               'cnf_model_ablation_out_dim_conf_18_0',
               'cnf_model_ablation_out_dim_conf_18_1',
               'cnf_model_ablation_out_dim_conf_18_2',
               'cnf_model_ablation_out_dim_conf_21_0',
               'cnf_model_ablation_out_dim_conf_21_1',
               'cnf_model_ablation_out_dim_conf_21_2'
          ]

for model in models:
    num_gen = 600
    general_path_generated = general_path + rf"\{model}_PCs"

    os.makedirs(path_tosave + fr'\{model}_PCs', exist_ok=True)

    pbar_gen = tqdm(total=num_gen, desc='Tagging generated anatomies...', leave=True)
    for i in range(num_gen):

        mesh = pv.read(general_path_generated + f'\PointCloud_{i}.vtk')
        mesh.cell_data['region_id'] = template_tagged.cell_data['region_id']

        # save
        mesh.save(path_tosave + rf'\{model}_PCs\PointCloud_{i}.vtk')

        pbar_gen.update()
    pbar_gen.close()











