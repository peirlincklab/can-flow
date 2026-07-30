import os
import shutil
from tqdm import tqdm



def copy2clean(model_str, all_num, model_str_path):

    os.makedirs(fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\Gen_Metrics_Exp_Files_Ablation\{model_str}_PCs", exist_ok=True)

    pbar = tqdm(total=all_num, desc='Copy PCs to clean folder...')
    for i in range(all_num):

        vtk_src_path = fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\{model_str_path}\Shooting_Momenta_{i}\output\Shooting__GeodesicFlow__biv__tp_10__age_1.00.vtk"

        new_folder_path =  fr"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\Gen_Metrics_Exp_Files_Ablation\{model_str}_PCs\PointCloud_{i}.vtk"

        # Ensure source exists
        if not os.path.exists(vtk_src_path):
            print("WARNING: source missing:", vtk_src_path)
            pbar.update()
            continue

        try:
            shutil.copy(src=vtk_src_path, dst=new_folder_path)
        except PermissionError as e:
            print("Permission error copying to", new_folder_path, "-", e)
        except Exception as e:
            print("Copy failed:", e)

        pbar.update()
    pbar.close()


### This is for ablation of CAN-FLOW neural architecture
# models = ['cnf_model_ablation_activation_elu_0',
#              'cnf_model_ablation_activation_elu_1',
#              'cnf_model_ablation_activation_elu_2',
#              'cnf_model_ablation_activation_leaky_relu_0',
#              'cnf_model_ablation_activation_leaky_relu_1',
#              'cnf_model_ablation_activation_leaky_relu_2',
#              'cnf_model_ablation_activation_relu_0',
#              'cnf_model_ablation_activation_relu_1',
#              'cnf_model_ablation_activation_relu_2',
#              'cnf_model_ablation_activation_silu_0',
#              'cnf_model_ablation_activation_silu_1',
#              'cnf_model_ablation_activation_silu_2',
#               'cnf_model_ablation_n_flow_5_0',
#               'cnf_model_ablation_n_flow_5_1',
#               'cnf_model_ablation_n_flow_5_2',
#               'cnf_model_ablation_n_flow_10_0',
#               'cnf_model_ablation_n_flow_10_1',
#               'cnf_model_ablation_n_flow_10_2',
#               'cnf_model_ablation_n_flow_20_0',
#               'cnf_model_ablation_n_flow_20_1',
#               'cnf_model_ablation_n_flow_20_2',
#               'cnf_model_ablation_n_flow_25_0',
#               'cnf_model_ablation_n_flow_25_1',
#               'cnf_model_ablation_n_flow_25_2',
#               'cnf_model_ablation_n_flow_30_0',
#               'cnf_model_ablation_n_flow_30_1',
#               'cnf_model_ablation_n_flow_30_2',
#                'cnf_model_ablation_out_dim_conf_6_0',
#                'cnf_model_ablation_out_dim_conf_6_1',
#                'cnf_model_ablation_out_dim_conf_6_2',
#                'cnf_model_ablation_out_dim_conf_9_0',
#                'cnf_model_ablation_out_dim_conf_9_1',
#                'cnf_model_ablation_out_dim_conf_9_2',
#                'cnf_model_ablation_out_dim_conf_15_0',
#                'cnf_model_ablation_out_dim_conf_15_1',
#                'cnf_model_ablation_out_dim_conf_15_2',
#                'cnf_model_ablation_out_dim_conf_18_0',
#                'cnf_model_ablation_out_dim_conf_18_1',
#                'cnf_model_ablation_out_dim_conf_18_2',
#                'cnf_model_ablation_out_dim_conf_21_0',
#                'cnf_model_ablation_out_dim_conf_21_1',
#                'cnf_model_ablation_out_dim_conf_21_2'
#           ]



### This is for ablation of latent dimensionality
models = ['AblationLatentDim_canflow_latent10',
          'AblationLatentDim_canflow_latent20',
          'AblationLatentDim_canflow_latent30',
          'AblationLatentDim_canflow_latent40',
          'AblationLatentDim_canflow_latent50',
          'AblationLatentDim_cvae2_latent10',
          'AblationLatentDim_cvae2_latent20',
          'AblationLatentDim_cvae2_latent30',
          'AblationLatentDim_cvae2_latent40',
          'AblationLatentDim_cvae2_latent50',
          'AblationLatentDim_cvae3_latent10',
          'AblationLatentDim_cvae3_latent20',
          'AblationLatentDim_cvae3_latent30',
          'AblationLatentDim_cvae3_latent40',
          'AblationLatentDim_cvae3_latent50']

os.makedirs( r"C:\Users\kkevopoulos\Documents\Meshes_Anatomies_Alternative_Branch\Ablation_anatomies\Gen_Metrics_Exp_Files_Ablation", exist_ok=True)

num_gen = 600 ### 300 female and 300 male synthetic anatomies for each model

for model in models:
    copy2clean(model_str=model, all_num=num_gen, model_str_path=f"{model}_Momenta")






