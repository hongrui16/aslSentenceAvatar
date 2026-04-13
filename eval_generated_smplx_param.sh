# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260226_223940_job6035473/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d \
#     --use_phono_attribute


# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260302_152358_job6201204/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d \
#     --use_phono_attribute \
#     --text_encoder_type t5



# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260310_030439_job6539103/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d \
#     --text_encoder_type t5


# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260313_170717_job6558069/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d 


# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260313_164658_job6558069/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d \
#     --text_encoder_type t5    


python eval_generated_smplx_param_v2.py \
    --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260226_023925_job6005717/best_model.pt \
    --render_mesh --gif --render_comparison \
    --use_upper_body --use_rot6d \
    --dataset_name ASL3DWord \
    --evaluate    